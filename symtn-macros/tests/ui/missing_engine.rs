use symtn_macros::contract;

fn main() {
    let _ = contract!([i; k] := a[i; j] * b[j; k]);
}
