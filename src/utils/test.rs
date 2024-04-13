macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {
        // log the values for debugging
        dbg!($a, $b);
        assert!(($a - $b).abs() < $eps);
    };
}
