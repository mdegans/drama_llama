#[cfg(test)]
#[macro_use]
pub mod test;

#[inline]
#[cold]
/// Marks a branch as unlikely.
pub(crate) fn cold() {}
