use symtn_macros::contract;

fn main() {
    let _ = contract!(&mut engine, [i; j] := a[i; j] + b[i; j]);
}
