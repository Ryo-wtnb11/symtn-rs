use symtn_macros::contract;

fn main() {
    let _ = contract!(&mut engine, [i] := a[i, j, j]);
}
