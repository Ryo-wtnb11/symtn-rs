# Mathematical Background

This page is the formula-first model behind symtn's user layer. The notation
is deliberately close to TensorKit's `TensorMap` convention; for the broader
motivation, see the TensorKit paper
[arXiv:2508.10076](https://arxiv.org/abs/2508.10076).

The short version:

- a [`crate::typed::GradedSpace`] is a finite direct sum of sector irreps with
  ordinary degeneracy spaces;
- a [`crate::typed::TensorMap`] is a morphism `codomain <- domain`;
- a domain leg is oriented as a dual object during contractions;
- leg rearrangements are morphisms in a braided rigid category;
- storage is one dense column-major matrix per coupled sector;
- norms and truncation budgets are weighted by quantum dimensions.

## Sectors And Spaces

Fix a multiplicity-free rigid fusion rule with simple sectors
`a, b, c, ...`, tensor unit `1`, dual sector `a*`, fusion coefficients
`N_ab^c`, and quantum dimensions `d_a`.

A symtn `GradedSpace` is one external tensor leg. Mathematically it is a finite
graded vector space

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>V</mi><mo>=</mo>
  <munder><mo>⊕</mo><mrow><mi>a</mi><mo>∈</mo><mi>ℐ</mi></mrow></munder>
  <msup><mi>𝕂</mi><msub><mi>m</mi><mi>a</mi></msub></msup>
  <mo>⊗</mo><mi>a</mi>
</math>
</div>

where `m_a` is the ordinary degeneracy attached to sector `a`. Its total
dimension is the quantum-dimension-weighted sum

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>dim</mi><mo stretchy="false">(</mo><mi>V</mi><mo stretchy="false">)</mo>
  <mo>=</mo>
  <munder><mo>∑</mo><mrow><mi>a</mi><mo>∈</mo><mi>ℐ</mi></mrow></munder>
  <msub><mi>m</mi><mi>a</mi></msub><msub><mi>d</mi><mi>a</mi></msub>
</math>
</div>

This is why

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>dim</mi><mo stretchy="false">(</mo><mtext>U(1)[−1 ↦ 2, 0 ↦ 3, 1 ↦ 2]</mtext><mo stretchy="false">)</mo>
  <mo>=</mo><mn>2</mn><mo>+</mo><mn>3</mn><mo>+</mo><mn>2</mn><mo>=</mo><mn>7</mn>
</math>
</div>

because every U(1) charge has quantum dimension `1`. For SU(2), symtn stores
`twice_spin = 2j`, and

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>d</mi><mi>j</mi></msub>
  <mo>=</mo><mn>2</mn><mi>j</mi><mo>+</mo><mn>1</mn>
  <mo>=</mo><mtext>twice_spin</mtext><mo>+</mo><mn>1</mn>
</math>
</div>

So an SU(2) `GradedSpace` with degeneracy two in twice-spin sectors 0 and 1 has dimension

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mn>2</mn><mo>·</mo><msub><mi>d</mi><mn>0</mn></msub>
  <mo>+</mo>
  <mn>2</mn><mo>·</mo><msub><mi>d</mi><mfrac><mn>1</mn><mn>2</mn></mfrac></msub>
  <mo>=</mo>
  <mn>2</mn><mo>·</mo><mn>1</mn><mo>+</mo><mn>2</mn><mo>·</mo><mn>2</mn>
  <mo>=</mo><mn>6</mn>
</math>
</div>

The user layer exposes U(1), bosonic Z2, fermion parity fZ2,
SU(2), and selected product symmetries through `GradedSpace` values over
admitted providers. Product-sector dimensions multiply sector
quantum dimensions componentwise.

## Duals

For an ordinary vector space, the dual is

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup><mi>V</mi><mo>*</mo></msup>
  <mo>=</mo>
  <mi>Hom</mi><mo stretchy="false">(</mo><mi>V</mi><mo>,</mo><mi>𝕂</mi><mo stretchy="false">)</mo>
</math>
</div>

If `e_i` is a basis for `V`, the dual basis `e^i` satisfies

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup><mi>e</mi><mi>i</mi></msup>
  <mo stretchy="false">(</mo><msub><mi>e</mi><mi>j</mi></msub><mo stretchy="false">)</mo>
  <mo>=</mo>
  <msubsup><mi>δ</mi><mi>j</mi><mi>i</mi></msubsup>
</math>
</div>

That evaluation pairing is the local operation behind index contraction.

For a graded space, duality acts on both the degeneracy space and the sector:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup><mi>V</mi><mo>*</mo></msup>
  <mo>=</mo>
  <munder><mo>⊕</mo><mrow><mi>a</mi><mo>∈</mo><mi>ℐ</mi></mrow></munder>
  <msup>
    <mrow><mo stretchy="false">(</mo><msup><mi>𝕂</mi><msub><mi>m</mi><mi>a</mi></msub></msup><mo stretchy="false">)</mo></mrow>
    <mo>*</mo>
  </msup>
  <mo>⊗</mo><msup><mi>a</mi><mo>*</mo></msup>
</math>
</div>

In symtn, [`crate::typed::GradedSpace::try_dual`] replaces every sector by its
fusion-rule dual and flips the leg's dual flag. For U(1), charge `q` dualizes
to `-q`; for Z2 and SU(2), the exposed sectors are self-dual, though SU(2)
still has nontrivial fusion and Frobenius-Schur data internally. Degeneracy
counts are not changed by dualization.

## Tensor Maps

symtn follows TensorKit's `codomain <- domain` convention. A tensor with
codomain legs `C_1, ..., C_m` and domain legs `D_1, ..., D_n` is a linear map

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd><mi>T</mi><mo>∈</mo><mi>Hom</mi><mo stretchy="false">(</mo><mi>D</mi><mo>,</mo><mi>C</mi><mo stretchy="false">)</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mi>C</mi><mo>=</mo><msub><mi>C</mi><mn>1</mn></msub><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msub><mi>C</mi><mi>m</mi></msub></mtd>
    </mtr>
    <mtr>
      <mtd><mi>D</mi><mo>=</mo><msub><mi>D</mi><mn>1</mn></msub><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msub><mi>D</mi><mi>n</mi></msub></mtd>
    </mtr>
  </mtable>
</math>
</div>

Equivalently, using the usual tensor-Hom identification,

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>Hom</mi><mo stretchy="false">(</mo><mi>D</mi><mo>,</mo><mi>C</mi><mo stretchy="false">)</mo>
  <mo>≅</mo><mi>C</mi><mo>⊗</mo><msup><mi>D</mi><mo>*</mo></msup>
</math>
</div>

This is the source of the orientation rule: codomain axes are objects, while
domain axes are dual objects. For a flat zero-based axis number `i`,

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>obj</mi><mi>T</mi></msub><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo>
  <mo>=</mo>
  <mrow>
    <mo>{</mo>
    <mtable columnspacing="1.35em" rowspacing="0.35em">
      <mtr>
        <mtd><msub><mi>C</mi><mi>i</mi></msub></mtd>
        <mtd><mrow><mn>0</mn><mo>≤</mo><mi>i</mi><mo>&lt;</mo><mi>m</mi></mrow></mtd>
      </mtr>
      <mtr>
        <mtd><msup><msub><mi>D</mi><mrow><mi>i</mi><mo>−</mo><mi>m</mi></mrow></msub><mo>*</mo></msup></mtd>
        <mtd><mrow><mi>m</mi><mo>≤</mo><mi>i</mi><mo>&lt;</mo><mi>m</mi><mo>+</mo><mi>n</mi></mrow></mtd>
      </mtr>
    </mtable>
  </mrow>
</math>
</div>

The stored `GradedSpace` on a domain side is still written as `D_i`; the dual
orientation appears when the tensor is interpreted as an element of
`C tensor D*` or when a contraction is planned.

## Contraction Compatibility

Two selected axes may be contracted exactly when their oriented objects are
dual:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>obj</mi><mi>A</mi></msub><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo>
  <mo>≅</mo>
  <msup>
    <mrow><msub><mi>obj</mi><mi>B</mi></msub><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow>
    <mo>*</mo>
  </msup>
</math>
</div>

The contraction itself is the evaluation morphism

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>ev</mi><mi>X</mi></msub>
  <mo>:</mo>
  <msup><mi>X</mi><mo>*</mo></msup><mo>⊗</mo><mi>X</mi>
  <mo>→</mo><mn>1</mn>
</math>
</div>

Ordinary composition is a special case. If

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd><mi>A</mi><mo>∈</mo><mi>Hom</mi><mo stretchy="false">(</mo><msub><mi>D</mi><mi>A</mi></msub><mo>,</mo><msub><mi>C</mi><mi>A</mi></msub><mo stretchy="false">)</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mi>B</mi><mo>∈</mo><mi>Hom</mi><mo stretchy="false">(</mo><msub><mi>D</mi><mi>B</mi></msub><mo>,</mo><msub><mi>C</mi><mi>B</mi></msub><mo stretchy="false">)</mo></mtd>
    </mtr>
    <mtr>
      <mtd><msub><mi>D</mi><mi>A</mi></msub><mo>≅</mo><msub><mi>C</mi><mi>B</mi></msub></mtd>
    </mtr>
  </mtable>
</math>
</div>

then

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>A</mi><mo>∘</mo><mi>B</mi>
  <mo>∈</mo>
  <mi>Hom</mi><mo stretchy="false">(</mo><msub><mi>D</mi><mi>B</mi></msub><mo>,</mo><msub><mi>C</mi><mi>A</mi></msub><mo stretchy="false">)</mo>
</math>
</div>

So "`one leg is on the domain side`" is not the general mathematical rule.
It is only what happens for the common codomain-vs-domain composition case.
Same-side contraction is valid when the actual `GradedSpace` on exactly one of the
two same-side legs is dualized:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnspacing="1.35em" rowspacing="0.45em">
    <mtr>
      <mtd><mtext>codomain </mtext><mi>V</mi><mtext> with domain </mtext><mi>V</mi></mtd>
      <mtd><mi>V</mi><mtext> pairs with </mtext><msup><mi>V</mi><mo>*</mo></msup></mtd>
    </mtr>
    <mtr>
      <mtd><mtext>domain </mtext><mi>V</mi><mtext> with domain </mtext><msup><mi>V</mi><mo>*</mo></msup></mtd>
      <mtd><msup><mi>V</mi><mo>*</mo></msup><mtext> pairs with </mtext><msup><mrow><mo stretchy="false">(</mo><msup><mi>V</mi><mo>*</mo></msup><mo stretchy="false">)</mo></mrow><mo>*</mo></msup><mo>≅</mo><mi>V</mi></mtd>
    </mtr>
    <mtr>
      <mtd><mtext>domain </mtext><mi>V</mi><mtext> with domain </mtext><mi>V</mi></mtd>
      <mtd><msup><mi>V</mi><mo>*</mo></msup><mtext> does not pair with </mtext><msup><mi>V</mi><mo>*</mo></msup></mtd>
    </mtr>
  </mtable>
</math>
</div>

For U(1), this says a charge `q` object pairs with a charge `-q` object after
orientation is accounted for. A `GradedSpace` need not contain a charge set closed
under negation; dualization constructs the corresponding dual sector set.

## Leg Re-Arrangements

The current user API exposes [`crate::typed::TensorMap::permute`],
[`crate::typed::TensorMap::braid`], [`crate::typed::TensorMap::transpose`],
[`crate::typed::TensorMap::adjoint`], [`crate::typed::TensorMap::flip`], and
[`crate::typed::TensorMap::twist`]. The latter two act on fusion-tree legs:
`flip` changes leg duality, while `twist` applies ribbon-twist phases.

Let

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd><mi>T</mi><mo>∈</mo><mi>Hom</mi><mo stretchy="false">(</mo><mi>D</mi><mo>,</mo><mi>C</mi><mo stretchy="false">)</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mi>C</mi><mo>=</mo><msub><mi>C</mi><mn>1</mn></msub><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msub><mi>C</mi><mi>m</mi></msub></mtd>
    </mtr>
    <mtr>
      <mtd><mi>D</mi><mo>=</mo><msub><mi>D</mi><mn>1</mn></msub><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msub><mi>D</mi><mi>n</mi></msub></mtd>
    </mtr>
  </mtable>
</math>
</div>

Choose two ordered axis lists `I` and `J` whose union is every flat source
axis exactly once. The transformed tensor has

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.45em">
    <mtr>
      <mtd>
        <mi>codomain</mi><mo stretchy="false">(</mo><msup><mi>T</mi><mo>′</mo></msup><mo stretchy="false">)</mo>
        <mo>=</mo>
        <munder><mo>⨂</mo><mrow><mi>i</mi><mo>∈</mo><mi>I</mi></mrow></munder>
        <msub><mi>obj</mi><mi>T</mi></msub><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <mi>domain</mi><mo stretchy="false">(</mo><msup><mi>T</mi><mo>′</mo></msup><mo stretchy="false">)</mo>
        <mo>=</mo>
        <munder><mo>⨂</mo><mrow><mi>j</mi><mo>∈</mo><mi>J</mi></mrow></munder>
        <msup>
          <mrow><msub><mi>obj</mi><mi>T</mi></msub><mo stretchy="false">(</mo><mi>j</mi><mo stretchy="false">)</mo></mrow>
          <mo>*</mo>
        </msup>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

The different operations are different categorical ways to realize this same
change of boundary order:

- `permute` uses the symmetric braiding, so any permutation of axes is
  allowed when the fusion rule has symmetric braiding.
- `braid` uses an explicit braid word; the result can depend on which strand
  crosses over, through the rule's `R` and `F` symbols.
- `transpose` is the planar move: it rotates the boundary without inserting
  arbitrary crossings.

For the built-in full transpose, the codomain and domain swap with duals and
reversed order:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd><msup><mi>T</mi><mi mathvariant="normal">T</mi></msup><mo>∈</mo><mi>Hom</mi><mo stretchy="false">(</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mspace width="1.5em"/><msup><msub><mi>C</mi><mi>m</mi></msub><mo>*</mo></msup><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msup><msub><mi>C</mi><mn>1</mn></msub><mo>*</mo></msup><mo>,</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mspace width="1.5em"/><msup><msub><mi>D</mi><mi>n</mi></msub><mo>*</mo></msup><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msup><msub><mi>D</mi><mn>1</mn></msub><mo>*</mo></msup><mo stretchy="false">)</mo></mtd>
    </mtr>
  </mtable>
</math>
</div>

The dagger is the categorical adjoint:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup><mi>T</mi><mi>†</mi></msup>
  <mo>∈</mo>
  <mi>Hom</mi><mo stretchy="false">(</mo><mi>C</mi><mo>,</mo><mi>D</mi><mo stretchy="false">)</mo>
  <mo>,</mo>
  <msub><mrow><mo>⟨</mo><mi>T</mi><mi>x</mi><mo>,</mo><mi>y</mi><mo>⟩</mo></mrow><mi>C</mi></msub>
  <mo>=</mo>
  <msub><mrow><mo>⟨</mo><mi>x</mi><mo>,</mo><msup><mi>T</mi><mi>†</mi></msup><mi>y</mi><mo>⟩</mo></mrow><mi>D</mi></msub>
</math>
</div>

For real `f64` scalar data, the scalar conjugation part of `dagger` is
invisible, but the codomain/domain swap and fusion-tree view change still
matter.

### Flip And Twist

TensorKit's `flip` changes the duality flag of one fusion-tree leg. It is not
just a boolean metadata update: the tensor is multiplied by the scalar from
the rigid-category Z-isomorphism that identifies an outgoing `a` line with an
incoming `a*` line.

For a leg carrying sector `a`, write `theta_a` for the topological twist and
`chi_a` for the Frobenius-Schur phase. If `epsilon` is the current dual flag
of that fusion-tree leg (`0` = not dual, `1` = dual), TensorKit's forward
single-leg flip uses the coefficients

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.75em">
    <mtr>
      <mtd>
        <msubsup><mi>z</mi><mi>a</mi><mi>out</mi></msubsup><mo stretchy="false">(</mo><mi>ε</mi><mo stretchy="false">)</mo>
        <mo>=</mo>
        <mrow>
          <mo>{</mo>
          <mtable columnspacing="1.1em" rowspacing="0.25em">
            <mtr><mtd><mn>1</mn></mtd><mtd><mi>ε</mi><mo>=</mo><mn>0</mn></mtd></mtr>
            <mtr><mtd><msub><mi>χ</mi><mi>a</mi></msub><mspace width="0.15em"/><msub><mi>θ</mi><mi>a</mi></msub></mtd><mtd><mi>ε</mi><mo>=</mo><mn>1</mn></mtd></mtr>
          </mtable>
        </mrow>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msubsup><mi>z</mi><mi>a</mi><mi>in</mi></msubsup><mo stretchy="false">(</mo><mi>ε</mi><mo stretchy="false">)</mo>
        <mo>=</mo>
        <mrow>
          <mo>{</mo>
          <mtable columnspacing="1.1em" rowspacing="0.25em">
            <mtr><mtd><msub><mi>θ</mi><mi>a</mi></msub></mtd><mtd><mi>ε</mi><mo>=</mo><mn>0</mn></mtd></mtr>
            <mtr><mtd><mover><msub><mi>χ</mi><mi>a</mi></msub><mo>¯</mo></mover></mtd><mtd><mi>ε</mi><mo>=</mo><mn>1</mn></mtd></mtr>
          </mtable>
        </mrow>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

and changes

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>ε</mi><mo>↦</mo><mn>1</mn><mo>−</mo><mi>ε</mi>
</math>
</div>

Here `out` means a codomain fusion-tree leg and `in` means a domain
fusion-tree leg. The inverse flip uses the inverse Z-isomorphism, so in a
general braided rigid category `flip` is not simply its own inverse; TensorKit
warns that only four flips are guaranteed to compose back to the identity in
full generality.

The related `twist` operation does not change the leg's duality flag. It
multiplies the corresponding fusion-tree component by the twist scalar:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub>
    <mrow><msub><mi>twist</mi><mi>i</mi></msub><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow>
    <mrow><mo>…</mo><mo>,</mo><mi>a</mi><mo>,</mo><mo>…</mo></mrow>
  </msub>
  <mo>=</mo>
  <msub><mi>θ</mi><mi>a</mi></msub>
  <msub><mi>T</mi><mrow><mo>…</mo><mo>,</mo><mi>a</mi><mo>,</mo><mo>…</mo></mrow></msub>
</math>
</div>

for the selected leg `i`, up to the side/orientation convention of that leg.
For bosonic U(1) and Z2 sectors this is usually `1`; for fermion parity, the
odd sector has twist `-1`.

## Fusion-Tree Basis And Blocks

For a product space `X = X_1 tensor ... tensor X_n`, symtn enumerates fusion
trees ending in a coupled sector `c`. Conceptually, these are basis labels for
spaces such as

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd><mi>Hom</mi><mo stretchy="false">(</mo><mi>c</mi><mo>,</mo><msub><mi>X</mi><mn>1</mn></msub><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msub><mi>X</mi><mi>n</mi></msub><mo stretchy="false">)</mo></mtd>
    </mtr>
    <mtr>
      <mtd><mtext>or</mtext></mtd>
    </mtr>
    <mtr>
      <mtd><mi>Hom</mi><mo stretchy="false">(</mo><msub><mi>X</mi><mn>1</mn></msub><mo>⊗</mo><mo>⋯</mo><mo>⊗</mo><msub><mi>X</mi><mi>n</mi></msub><mo>,</mo><mi>c</mi><mo stretchy="false">)</mo></mtd>
    </mtr>
  </mtable>
</math>
</div>

together with the degeneracy indices on each external leg.

For a tensor map `T : D -> C`, codomain and domain fusion trees are paired
only when they have the same coupled sector. Thus the tensor decomposes as
one matrix per coupled sector:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd>
        <mi>T</mi><mo>↔</mo>
        <msub>
          <mrow><mo>{</mo><msub><mi>T</mi><mi>c</mi></msub><mo>}</mo></mrow>
          <mrow><mi>c</mi><mo>∈</mo><mi>ℐ</mi></mrow>
        </msub>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub><mi>T</mi><mi>c</mi></msub>
        <mo>∈</mo>
        <msup><mi>𝕂</mi><mrow><msub><mi>r</mi><mi>C</mi></msub><mo stretchy="false">(</mo><mi>c</mi><mo stretchy="false">)</mo><mo>×</mo><msub><mi>r</mi><mi>D</mi></msub><mo stretchy="false">(</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></msup>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

Here `r_C(c)` is the total row count from codomain fusion trees ending in
`c` times their codomain degeneracies, and `r_D(c)` is the analogous domain
column count. The actual `BlockStructure` stores those sector matrices in
column-major order; individual fusion-tree subblocks are strided views into
the sector matrix.

This is the layout of every [`crate::typed::TensorMap`]. The
user-layer [`crate::typed::TensorMap::dense_data`] method borrows the same flat
storage.

## Inner Products, Norms, And Truncation

The Frobenius inner product is weighted by the quantum dimension of the
coupled sector:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mo>⟨</mo><mi>A</mi><mo>,</mo><mi>B</mi><mo>⟩</mo>
  <mo>=</mo>
  <munder><mo>∑</mo><mrow><mi>c</mi><mo>∈</mo><mi>ℐ</mi></mrow></munder>
  <msub><mi>d</mi><mi>c</mi></msub>
  <mi>tr</mi><mo stretchy="false">(</mo><msubsup><mi>A</mi><mi>c</mi><mi>†</mi></msubsup><msub><mi>B</mi><mi>c</mi></msub><mo stretchy="false">)</mo>
</math>
</div>

Consequently,

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msup><mrow><mo>∥</mo><mi>A</mi><mo>∥</mo></mrow><mn>2</mn></msup>
  <mo>=</mo>
  <munder><mo>∑</mo><mrow><mi>c</mi><mo>∈</mo><mi>ℐ</mi></mrow></munder>
  <msub><mi>d</mi><mi>c</mi></msub>
  <msubsup><mrow><mo>∥</mo><msub><mi>A</mi><mi>c</mi></msub><mo>∥</mo></mrow><mi>F</mi><mn>2</mn></msubsup>
</math>
</div>

This is the norm used by [`crate::typed::TensorMap::norm`] and
[`crate::typed::TensorMap::inner`]. It is also the norm used by SVD and
Hermitian eigentruncation errors.

For a sectorwise SVD,

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <msub><mi>T</mi><mi>c</mi></msub>
  <mo>=</mo>
  <msub><mi>U</mi><mi>c</mi></msub><msub><mi>S</mi><mi>c</mi></msub><msubsup><mi>V</mi><mi>c</mi><mi>†</mi></msubsup>
</math>
</div>

discarded singular values contribute

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.35em">
    <mtr>
      <mtd>
        <msup><mi>ε</mi><mn>2</mn></msup>
        <mo>=</mo>
        <munder><mo>∑</mo><mi>c</mi></munder>
        <msub><mi>d</mi><mi>c</mi></msub>
        <munder><mo>∑</mo><mrow><mi>σ</mi><mo>∈</mo><mi>discarded</mi><mo stretchy="false">(</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></munder>
        <msup><mi>σ</mi><mn>2</mn></msup>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

`find_truncated` reports `epsilon` as its `error`, so after an SVD truncated
with its selection

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mo>∥</mo><mi>T</mi><mo>−</mo><mi>U</mi><mi>S</mi><msup><mi>V</mi><mi>†</mi></msup><mo>∥</mo>
  <mo>=</mo><mi>ε</mi>
</math>
</div>

Likewise, a rank budget is a quantum-dimension budget:

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtext>weighted rank</mtext>
  <mo>=</mo>
  <munder><mo>∑</mo><mi>c</mi></munder>
  <msub><mi>d</mi><mi>c</mi></msub><msub><mi>k</mi><mi>c</mi></msub>
</math>
</div>

where `k_c` is the number of kept singular values in coupled sector `c`.
For SU(2), keeping one spin-`j` singular vector consumes `2j + 1` units of
rank budget.

## Operations In This Language

- `compose` is categorical composition: match the domain of the left tensor
  with the codomain of the right tensor.
- `contract` and `symtn_network::Network` lower arbitrary repeated labels to the same
  dual-pairing rule, then choose a pairwise execution plan.
- `permute`, `braid`, and `transpose` change the fusion-tree basis as well as
  the apparent axis order. With fermion parity and SU(2), this can introduce
  rule-dependent signs or basis transforms.
- `adjoint` swaps codomain and domain and conjugates the scalar data.
- TensorKit-style `flip` toggles a leg's duality flag and multiplies by the
  relevant Z-isomorphism coefficient; `twist` multiplies by the sector twist.
- `svd_compact`, `eigh_full`, and related decompositions operate blockwise on
  the coupled-sector matrices. Truncating one is a separate,
  quantum-dimension-weighted decision over all sectors
  (`GradedSpace::find_truncated`), applied with `restrict_leg` (on both
  legs of the spectrum factor at once).
- `left_null` and `right_null` are shape-based, as TensorKit's default
  (`qr_null`/`lq_null`): in sector `c` the left null space of an
  `m_c × n_c` block has dimension `(m_c − n_c)₊`, the orthogonal complement
  of a superspace of the block's column space, whatever its rank. The
  rank-revealing null space (TensorKit `left_null(t; alg = :svd, trunc)`) is
  a composition: `svd_full`, the diagonal of its `s` padded with the
  rectangular zeros to `u`'s bond (`GradedSpace::extend_spectrum`), the
  keep-below selection `Truncation::below(atol, rtol)` — the states whose
  singular value is at most `max(atol, rtol ∥S∥)`, `∥S∥` the
  quantum-dimension-weighted norm above — optionally intersected with
  `Truncation::rank_smallest(maxnullity)`, a weighted budget filled from the
  smallest values, and `restrict_leg` on `u`. Sector `c` then keeps
  `(m_c − n_c)₊` rectangular states plus one per small singular value.
- `restrict_leg` composes with the inclusion isometry of a `LegSelection`
  `σ = { c ↦ P_c }`, a set of kept degeneracy positions
  `P_c = {p_0 < … < p_{k_c − 1}}` per sector: the identity on every irrep
  and the order-preserving coordinate inclusion `e_j ↦ e_{p_j}` on the
  degeneracy space, so whole multiplets are kept or dropped and the kept
  states stay in stored order. `embed_leg` is its adjoint (zeros at the
  positions not kept). A contiguous range is one such set; the kept
  largest-`|λ|` states of an ascending `eigh` spectrum or an arbitrary `eig`
  set are others.

The method `exp` is the sectorwise matrix exponential. For a Hermitian
endomorphism `H`, an imaginary-time gate uses

<div class="math" style="margin: 1.25rem 0; padding: 0.2rem 0; overflow-x: auto;">
<math display="block" style="font-size: 1.12em; line-height: 1.8;" xmlns="http://www.w3.org/1998/Math/MathML">
  <mtable columnalign="left" rowspacing="0.45em">
    <mtr>
      <mtd>
        <mi>G</mi><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo>
        <mo>=</mo><mi mathvariant="normal">exp</mi><mo stretchy="false">(</mo><mo>−</mo><mi>τ</mi><mspace width="0.2em"/><mi>H</mi><mo stretchy="false">)</mo>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub><mi>H</mi><mi>c</mi></msub>
        <mo>=</mo>
        <msub><mi>V</mi><mi>c</mi></msub><mspace width="0.2em"/><msub><mi>Λ</mi><mi>c</mi></msub><mspace width="0.2em"/><msubsup><mi>V</mi><mi>c</mi><mi>†</mi></msubsup>
      </mtd>
    </mtr>
    <mtr>
      <mtd>
        <msub><mi>G</mi><mi>c</mi></msub><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo>
        <mo>=</mo>
        <msub><mi>V</mi><mi>c</mi></msub><mspace width="0.2em"/>
        <mi mathvariant="normal">exp</mi><mo stretchy="false">(</mo><mo>−</mo><mi>τ</mi><msub><mi>Λ</mi><mi>c</mi></msub><mo stretchy="false">)</mo>
        <mspace width="0.2em"/><msubsup><mi>V</mi><mi>c</mi><mi>†</mi></msubsup>
      </mtd>
    </mtr>
  </mtable>
</math>
</div>

## Current Scope

This page describes the model implemented by symtn today, not the full
generality of category-theoretic tensor calculus. Host ordinary operations are
multiplicity-free plus the checked Generic operations that state their own
bounds. `TableRule` has `Complex64` categorical coefficients and is
tested for construction, transformations including explicit planar braids,
tensor products, and composition. Ordinary arbitrary-axis contraction,
factorization, and general network execution are not supported or claimed.
Checked Generic support is Host-only. CUDA requires explicit transfer and
supports a multiplicity-free `f64` subset; unsupported device
operations return an explicit error.
