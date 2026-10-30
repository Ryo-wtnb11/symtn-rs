use symtn_macros::contract;

fn main() {
    let _ = contract!(&mut engine, [i; k] := a[i; j] * b[j; k] * c[j]);
}
