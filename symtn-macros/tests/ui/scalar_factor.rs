use symtn_macros::contract;

fn main() {
    let _ = contract!(&mut engine, [i; j] := 2.0 * a[i; j]);
}
