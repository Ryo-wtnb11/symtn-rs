#![forbid(unsafe_code)]

//! The `contract!` index-notation macro. Use it through its re-export
//! `symtn_network::contract!`, which documents the notation; the expansion
//! names `::symtn_network` by absolute path.

use std::collections::HashMap;

use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::{
    bracketed, parenthesized, token, Error, Expr, Ident, Index, Lit, LitFloat, Member, Result,
    Token,
};

/// Contracts a tensor network written in index notation; see
/// `symtn_network::contract!`.
#[proc_macro]
pub fn contract(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand(input.into())
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

/// A written label list: every label, and the `;` position if one was written.
struct Labels {
    labels: Vec<Ident>,
    split: Option<usize>,
}

struct Operand {
    tensor: TokenStream,
    conj: bool,
    labels: Labels,
}

struct Contraction {
    /// The caller's engine expression, evaluated once.
    engine: Expr,
    output: Option<Labels>,
    operands: Vec<Operand>,
}

impl Parse for Labels {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut labels = Vec::new();
        let mut split = None;
        while !input.is_empty() {
            if input.peek(Token![;]) {
                let semi: Token![;] = input.parse()?;
                if split.is_some() {
                    return Err(Error::new_spanned(semi, "a label list has at most one `;`"));
                }
                split = Some(labels.len());
            } else if input.peek(Ident::peek_any) {
                labels.push(input.call(Ident::parse_any)?);
                if input.peek(Token![,]) {
                    input.parse::<Token![,]>()?;
                }
            } else {
                return Err(input.error(
                    "a label is an identifier; labels are separated by `,` or spaces, \
                     and `;` separates codomain from domain",
                ));
            }
        }
        Ok(Self { labels, split })
    }
}

fn scalar_factor(span: Span) -> Error {
    Error::new(
        span,
        "contract! does not take scalar factors; \
         hint: scale the result with `TensorMap::scale`",
    )
}

impl Parse for Operand {
    fn parse(input: ParseStream) -> Result<Self> {
        if input.peek(Lit) || input.peek(Token![-]) {
            return Err(scalar_factor(input.span()));
        }
        let start = input.span();
        let (tensor, conj) = if input.peek(Ident) && input.peek2(token::Paren) && {
            let fork = input.fork();
            fork.parse::<Ident>()? == "conj"
        } {
            input.parse::<Ident>()?;
            let content;
            parenthesized!(content in input);
            (content.parse::<Expr>()?.into_token_stream(), true)
        } else if input.peek(token::Paren) {
            let content;
            parenthesized!(content in input);
            (content.parse::<Expr>()?.into_token_stream(), false)
        } else if input.peek(Ident::peek_any) {
            let mut tensor = input.call(Ident::parse_any)?.into_token_stream();
            while input.peek(Token![.]) {
                let dot: Token![.] = input.parse()?;
                if input.peek(LitFloat) {
                    // `t.0.0` lexes as `t`, `.`, the float `0.0`: two tuple
                    // indices, as rustc reads it.
                    let float: LitFloat = input.parse()?;
                    let indices = float
                        .to_string()
                        .split('.')
                        .map(|index| index.parse::<u32>().ok())
                        .collect::<Option<Vec<_>>>()
                        .filter(|indices| indices.len() == 2)
                        .ok_or_else(|| Error::new_spanned(&float, "expected a tuple index"))?;
                    let (first, second) = (
                        Index {
                            index: indices[0],
                            span: float.span(),
                        },
                        Index {
                            index: indices[1],
                            span: float.span(),
                        },
                    );
                    tensor.extend(quote!(#dot #first . #second));
                } else {
                    let member: Member = input.parse()?;
                    tensor.extend(quote!(#dot #member));
                }
            }
            if input.peek(token::Paren) || input.peek(Token![::]) {
                return Err(input.error(
                    "an operand is an identifier or a field chain; \
                     hint: put any other expression in parentheses, `(expr)[i; j]`",
                ));
            }
            (tensor, false)
        } else {
            return Err(input.error("expected an operand such as `a[i; j]`"));
        };
        if !input.peek(token::Bracket) {
            return Err(scalar_factor(start));
        }
        let content;
        bracketed!(content in input);
        if input.peek(token::Bracket) {
            return Err(Error::new(
                start,
                format!(
                    "an indexed expression is not an operand; hint: put it in \
                     parentheses, `({tensor}[{content}])[i; j]`"
                ),
            ));
        }
        Ok(Self {
            tensor,
            conj,
            labels: content.parse()?,
        })
    }
}

impl Parse for Contraction {
    fn parse(input: ParseStream) -> Result<Self> {
        let missing_engine = "contract! takes the caller's engine first; \
             hint: write `contract!(&mut engine, [i; k] := …)`";
        if input.peek(token::Bracket) {
            return Err(input.error(missing_engine));
        }
        let engine: Expr = input
            .parse()
            .map_err(|error| Error::new(error.span(), missing_engine))?;
        if !input.peek(Token![,]) {
            return Err(input.error(missing_engine));
        }
        input.parse::<Token![,]>()?;
        let output = if input.peek(token::Bracket) {
            let content;
            bracketed!(content in input);
            let output: Labels = content.parse()?;
            if input.peek(Token![+=]) {
                return Err(input.error(
                    "`+=` is not supported: a Network has no destination API; \
                     hint: create a new tensor with `:=`",
                ));
            }
            if input.peek(Token![:]) && input.peek2(Token![=]) {
                input.parse::<Token![:]>()?;
                input.parse::<Token![=]>()?;
            } else if input.peek(Token![=]) {
                return Err(input.error(
                    "`=` is not supported: a Network has no destination API; \
                     hint: create a new tensor with `:=`",
                ));
            } else {
                return Err(input.error("expected `:=` after the left-hand side"));
            }
            Some(output)
        } else {
            None
        };
        let mut operands = vec![input.parse::<Operand>()?];
        if output.is_none()
            && (input.peek(Token![:]) || input.peek(Token![=]) || input.peek(Token![+=]))
        {
            let name = &operands[0].tensor;
            return Err(input.error(format!(
                "the output of contract! has no name; hint: write \
                 `let {name} = contract!(&mut engine, [i; k] := …)?`"
            )));
        }
        while !input.is_empty() {
            if input.peek(Token![*]) {
                input.parse::<Token![*]>()?;
                operands.push(input.parse()?);
            } else if input.peek(Token![+]) || input.peek(Token![-]) {
                return Err(input.error(
                    "contract! contracts one product, not a sum; \
                     hint: contract each term and add them with `TensorMap::axpby`",
                ));
            } else {
                return Err(input.error("expected `*` between operands"));
            }
        }
        Ok(Self {
            engine,
            output,
            operands,
        })
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Site {
    Operand(usize),
    Output,
}

/// Rejects every label structure `Network::new` rejects, so that the
/// expansion's runtime construction cannot fail.
fn validate(contraction: &Contraction) -> Result<()> {
    // Each label's occurrences, in order of first appearance.
    let mut order: Vec<String> = Vec::new();
    let mut sites = HashMap::<String, Vec<(Site, &Ident)>>::new();
    let operands = contraction
        .operands
        .iter()
        .enumerate()
        .map(|(i, operand)| (Site::Operand(i), &operand.labels));
    let output = contraction
        .output
        .iter()
        .map(|labels| (Site::Output, labels));
    for (site, labels) in operands.chain(output) {
        for label in &labels.labels {
            let key = label.unraw().to_string();
            let occurrences = sites.entry(key.clone()).or_insert_with(|| {
                order.push(key);
                Vec::new()
            });
            if site != Site::Output && occurrences.iter().any(|&(s, _)| s == site) {
                return Err(Error::new_spanned(
                    label,
                    format!(
                        "label `{label}` is repeated within one operand (an intra-operand \
                         trace), which contract! does not take; hint: trace it first with \
                         `TensorMap::trace_pairs`"
                    ),
                ));
            }
            occurrences.push((site, label));
        }
    }
    let scalar = contraction.output.is_none();
    for key in &order {
        let occurrences = &sites[key];
        let (_, first) = occurrences[0];
        match occurrences.as_slice() {
            [_, _, (_, third), ..] => {
                return Err(Error::new_spanned(
                    third,
                    format!(
                        "label `{third}` appears {} times (a hyperedge); a label joins \
                         exactly two legs: two operand legs (contracted) or one operand leg \
                         and the left-hand side (open), and a Network has no \
                         hyperedges; hint: give each contracted leg pair its own label",
                        occurrences.len()
                    ),
                ));
            }
            [(Site::Output, _), (Site::Output, second)] => {
                return Err(Error::new_spanned(
                    second,
                    format!("label `{second}` is repeated on the left-hand side"),
                ));
            }
            [(Site::Output, _)] => {
                return Err(Error::new_spanned(
                    first,
                    format!("left-hand-side label `{first}` appears on no operand"),
                ));
            }
            [_] if scalar => {
                return Err(Error::new_spanned(
                    first,
                    format!(
                        "label `{first}` is open, but contract! without a left-hand side \
                         contracts every label to a scalar; hint: pair the label, or \
                         write the open labels as `[..] :=`"
                    ),
                ));
            }
            [_] => {
                return Err(Error::new_spanned(
                    first,
                    format!(
                        "label `{first}` appears on one operand only: write it on the \
                         left-hand side (single-operand sums are not supported)"
                    ),
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn label_strings(labels: &Labels) -> TokenStream {
    let names = labels.labels.iter().map(|label| label.unraw().to_string());
    quote!(::std::vec::Vec::<&'static str>::from([#(#names),*]))
}

fn expand(input: TokenStream) -> Result<TokenStream> {
    let contraction: Contraction = syn::parse2(input)?;
    validate(&contraction)?;
    let inputs = contraction
        .operands
        .iter()
        .map(|operand| label_strings(&operand.labels));
    let conj = contraction.operands.iter().map(|operand| operand.conj);
    // An operand written without `;` has every leg in the codomain.
    let splits = contraction.operands.iter().map(|operand| {
        let labels = &operand.labels;
        labels.split.unwrap_or(labels.labels.len())
    });
    let tensors = contraction.operands.iter().map(|operand| &operand.tensor);
    let engine = &contraction.engine;
    let (output, output_split) = match &contraction.output {
        Some(labels) => (label_strings(labels), labels.split),
        None => (quote!(::std::vec::Vec::<&'static str>::new()), None),
    };
    let output_split = match output_split {
        Some(split) => quote!(::core::option::Option::Some(#split)),
        None => quote!(::core::option::Option::None),
    };
    // No left-hand side: the rank-0 result's single element.
    let scalar = contraction
        .output
        .is_none()
        .then(|| quote!(.and_then(|scalar| scalar.scalar(__symtn_engine).map_err(::core::convert::Into::into))));
    Ok(quote! {
        {
            use ::symtn_network::NetworkOperand as _;
            // Reborrowed, so a `&mut dyn Engine` binding stays usable after
            // the call.
            let __symtn_engine = &mut *(#engine);
            static __SYMTN_CONTRACT_NETWORK: ::std::sync::OnceLock<::symtn_network::Network> =
                ::std::sync::OnceLock::new();
            __SYMTN_CONTRACT_NETWORK
                .get_or_init(|| {
                    ::symtn_network::Network::new(
                        [#(#inputs),*],
                        [#(#conj),*],
                        [#(#splits),*],
                        #output,
                        #output_split,
                    )
                    // contract! rejected at compile time every label
                    // structure Network::new rejects (traces, hyperedges,
                    // unpaired or unknown labels), and the splits and
                    // output rank are derived from the written lists.
                    .expect("contract! validated its label structure at compile time")
                })
                .contract(__symtn_engine, &[#((#tensors).__contract_operand()),*])
                #scalar
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn error(input: TokenStream) -> String {
        match expand(input) {
            Ok(_) => panic!("expected a compile error"),
            Err(error) => error.to_string(),
        }
    }

    /// The `Network::new` arguments of an expansion, whitespace-free.
    fn network_args(input: TokenStream) -> String {
        let expanded = expand(input).unwrap().to_string().replace(' ', "");
        let start = expanded.find("Network::new(").unwrap() + "Network::new(".len();
        let end = expanded.find(").expect(").unwrap();
        expanded[start..end].to_owned()
    }

    #[test]
    fn each_written_form_lowers_to_its_network_arguments() {
        let vec = "::std::vec::Vec::<&'staticstr>::";
        assert_eq!(
            network_args(quote!(&mut engine, [i; k] := a[i; j] * b[j; k])),
            format!(
                "[{vec}from([\"i\",\"j\"]),{vec}from([\"j\",\"k\"])],[false,false],\
                 [1usize,1usize],{vec}from([\"i\",\"k\"]),::core::option::Option::Some(1usize),"
            )
        );
        // No `;`: every leg in the codomain; labels separated by spaces.
        assert_eq!(
            network_args(quote!(&mut engine, [a b c] := x[a i] * y[i; b, c])),
            format!(
                "[{vec}from([\"a\",\"i\"]),{vec}from([\"i\",\"b\",\"c\"])],[false,false],\
                 [2usize,1usize],{vec}from([\"a\",\"b\",\"c\"]),::core::option::Option::None,"
            )
        );
        // No left-hand side: an empty output and the scalar accessor.
        let scalar = quote!(&mut engine, conj(psi)[p; l, r] * h[p; q] * psi[q; l, r]);
        assert_eq!(
            network_args(scalar.clone()),
            format!(
                "[{vec}from([\"p\",\"l\",\"r\"]),{vec}from([\"p\",\"q\"]),\
                 {vec}from([\"q\",\"l\",\"r\"])],[true,false,false],[1usize,1usize,1usize],\
                 {vec}new(),::core::option::Option::None,"
            )
        );
        assert!(expand(scalar)
            .unwrap()
            .to_string()
            .contains("scalar (__symtn_engine)"));
        let rank0 = expand(quote!(&mut engine, [] := a[i; j] * b[j; i]))
            .unwrap()
            .to_string();
        assert!(!rank0.contains("scalar (__symtn_engine)"));
    }

    #[test]
    fn operands_are_identifiers_field_chains_or_parenthesized_expressions() {
        let expanded =
            expand(quote!(&mut engine, [i; k] := (svd.u)[i; s] * conj(w.get(0)?)[k; s] * x.0.y[]))
                .unwrap()
                .to_string()
                .replace(' ', "");
        assert!(expanded.contains(
            ".contract(__symtn_engine,&[(svd.u).__contract_operand(),\
             (w.get(0)?).__contract_operand(),(x.0.y).__contract_operand()])"
        ));
        assert!(network_args(quote!(&mut engine, [r#in] := r#type[r#in])).contains("\"in\""));
        let nested = expand(quote!(&mut engine, [i, k] := tup.0.0[i; j] * pair.1.2.u[j; k]))
            .unwrap()
            .to_string()
            .replace(' ', "");
        assert!(nested.contains("(tup.0.0).__contract_operand(),(pair.1.2.u).__contract_operand()"));
    }

    #[test]
    fn rejected_forms_name_their_hint() {
        let cases = [
            (quote!(&mut engine, [i] := a[i, j, j]), "trace_pairs"),
            (quote!(&mut engine, [] := a[i] * b[i] * c[i]), "hyperedge"),
            (
                quote!(&mut engine, [i] := a[i; j] * b[j] * c[i]),
                "hyperedge",
            ),
            (quote!(&mut engine, [i] := a[i] + b[i]), "axpby"),
            (quote!(&mut engine, [i] := a[i] - b[i]), "axpby"),
            (quote!(&mut engine, [i] := 2.0 * a[i]), "scale"),
            (quote!(&mut engine, [i] := alpha * a[i]), "scale"),
            (quote!(&mut engine, [i] := -a[i]), "scale"),
            (quote!(&mut engine, [i] := (s)), "scale"),
            (quote!(&mut engine, [i] = a[i]), "destination"),
            (quote!(&mut engine, [i] += a[i]), "destination"),
            (quote!(&mut engine, [i] := a.f(0)[i]), "parentheses"),
            (quote!(&mut engine, [i, i] := a[i] * b[j, j]), "trace_pairs"),
            (
                quote!(&mut engine, [i, i] := a[j] * b[j]),
                "repeated on the left-hand side",
            ),
            (quote!(&mut engine, [i, k] := a[i]), "appears on no operand"),
            (
                quote!(&mut engine, [] := a[i, j] * b[j]),
                "one operand only",
            ),
            (
                quote!(&mut engine, a[i, j] * b[j]),
                "contracts every label to a scalar",
            ),
            (quote!(&mut engine, [i] := a[i; ; j]), "at most one `;`"),
            (quote!(&mut engine, [i] := a[1]), "a label is an identifier"),
            (quote!(&mut engine, [i] - a[i]), "expected `:=`"),
            (quote!(&mut engine, [i] := a[i] / b[i]), "expected `*`"),
            (quote!(&mut engine, [i] := * a[i]), "expected an operand"),
            (quote!(&mut engine, [i; j] := w[0][i; j]), "`(w[0])[i; j]`"),
            (
                quote!(&mut engine, c[i; k] := a[i; j] * b[j; k]),
                "`let c = contract!(&mut engine, [i; k] := …)?`",
            ),
            (
                quote!(&mut engine, c[i; k] = a[i; j] * b[j; k]),
                "has no name",
            ),
            (
                quote!([i; k] := a[i; j] * b[j; k]),
                "takes the caller's engine first",
            ),
            (
                quote!(&mut engine [i] := a[i]),
                "takes the caller's engine first",
            ),
        ];
        for (input, hint) in cases {
            let message = error(input.clone());
            assert!(message.contains(hint), "{input}: {message}");
        }
    }
}
