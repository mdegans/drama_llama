#[cfg(feature = "serde")]
use rocket::serde::Deserialize;

/// Error for invalid probability values.
#[derive(Debug, PartialEq, thiserror::Error)]
#[error("Invalid probability. Must be between 0.0 and 1.0. Got {p}")]
pub struct InvalidProbability<F>
where
    F: std::fmt::Display,
{
    p: F,
}

/// A [`Probability`] is a wrapper around a floating point number that
/// represents a probability. It is guaranteed to be between 0.0 and 1.0.
#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Hash)]
#[repr(transparent)]
pub struct Probability<F> {
    p: F,
}
impl<F> Probability<F> {
    pub fn from_f(p: F) -> Result<Self, InvalidProbability<F>>
    where
        F: num::Zero + num::One + std::cmp::PartialOrd + std::fmt::Display,
    {
        if p >= F::zero() && p <= F::one() {
            Ok(Probability { p })
        } else {
            Err(InvalidProbability { p })
        }
    }

    pub fn into_f(self) -> F {
        self.p
    }
}

impl<F> PartialEq<F> for Probability<F>
where
    F: PartialEq<F>,
{
    fn eq(&self, other: &F) -> bool {
        self.p.eq(other)
    }
}

impl<F> PartialOrd<F> for Probability<F>
where
    F: PartialOrd<F>,
{
    fn partial_cmp(&self, other: &F) -> Option<std::cmp::Ordering> {
        self.p.partial_cmp(other)
    }
}

#[cfg(feature = "serde")]
impl<'de, F> Deserialize<'de> for Probability<F>
where
    F: Deserialize<'de>
        + num::Zero
        + num::One
        + std::cmp::PartialOrd
        + std::fmt::Display,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: rocket::serde::Deserializer<'de>,
    {
        let p = F::deserialize(deserializer)?;
        Probability::from_f(p).map_err(|e| rocket::serde::de::Error::custom(e))
    }
}

#[cfg(feature = "serde")]
impl<F> rocket::serde::Serialize for Probability<F>
where
    F: rocket::serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: rocket::serde::Serializer,
    {
        self.p.serialize(serializer)
    }
}

// Rust complains about conflicting implementations of the conversion trait for
// the same type, so we need to use a macro to generate the impls.
macro_rules! impl_from_to_float {
    ($($t:ty),*) => {
        $(
            impl TryFrom<$t> for Probability<$t> {
                type Error = InvalidProbability<$t>;

                fn try_from(p: $t) -> Result<Self, Self::Error> {
                    Probability::from_f(p)
                }
            }

            impl Into<$t> for Probability<$t> {
                fn into(self) -> $t {
                    self.into_f()
                }
            }
        )*
    };
    () => {};
}

impl_from_to_float!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability() {
        // Probabilities are invalid if out of bounds
        let err = Probability::try_from(1.1_f64).unwrap_err();
        assert_eq!(err.p, 1.1);
        let err = Probability::try_from(-0.1_f32).unwrap_err();
        assert_eq!(err.p, -0.1);

        // Test valid probability
        let p = Probability::try_from(0.5).unwrap();
        assert_eq!(p, 0.5);

        // Test comparison with F
        assert!(p > 0.0 && p < 1.0);

        // Test conversion to float
        let f: f32 = p.into();
        assert_eq!(f, 0.5);
    }
}
