use llama_cpp_sys_3::{
    llama_batch, llama_batch_free, llama_batch_init, llama_seq_id, llama_token,
};
use thiserror::Error;

/// A `Batch` of tokens or embeddings. This wraps a [`llama_batch`] and provides
/// safe accessors for it's members.
#[derive(Debug)]
pub struct Batch {
    /// The underlying C struct
    pub(crate) batch: llama_batch,
    /// Batch size (maximum number of members in the batch)
    pub(crate) capacity: usize,
    /// The number of allocated embeddings. When batch.tokens is null:
    ///
    /// ```C
    /// batch.embd = (float *) malloc(sizeof(float) * capacity * embd);
    /// ```
    pub(crate) embd_len: usize,
    /// The maximum number of sequence ids per token.
    pub(crate) n_seq_max: usize,
}

#[derive(Debug, Error, PartialEq)]
pub enum AddError {
    #[error("The batch is full")]
    Full,
    #[error("The number of sequence ids does not match the batch's n_seq_max")]
    InvalidSequenceLength,
    // FIXME: add `add_embedding` method to `Batch`
    #[error("A token was supplied, but thet batch was created with embd_len > 0. Call `add_embedding` instead.")]
    ExpectedEmbedding,
    #[error("An embedding was supplied, but thet batch was created with embd_len == 0. Call `add_token` instead.")]
    ExpectedToken,
    #[error("Invalid token position.")]
    InvalidPosition,
}

impl Batch {
    /// Create a new [`Batch`] with the given `capacity` for tokens or
    /// embeddings. If `embd_len` is zero, the `tokens` accessor will be
    /// available, otherwise the `embd` accessor will be available. Each token
    /// can be assigned up to `n_seq_max` sequence ids.
    pub fn new(
        capacity: usize,
        embd_len: usize,
        n_seq_max: usize,
    ) -> Option<Self> {
        let batch = unsafe {
            llama_batch_init(
                capacity.try_into().ok()?,
                embd_len.try_into().ok()?,
                n_seq_max.try_into().ok()?,
            )
        };

        // sanity
        debug_assert!(batch.n_tokens == 0);

        Some(Self {
            batch,
            capacity,
            embd_len,
            n_seq_max,
        })
    }

    /// Create a new [`Batch`] with capacity for tokens. The the `logit` field
    /// for all but the last token will be set to `false`. If the capacity is
    /// less than the number of tokens, the largest value will be used.
    pub fn from_tokens(
        capacity: usize,
        tokens: &[llama_token],
    ) -> Option<Self> {
        let mut batch = Self::new(capacity.max(tokens.len()), 0, 1)?;

        for (i, token) in tokens.iter().enumerate() {
            let logits = i == tokens.len() - 1;
            batch.add_token(*token, i, None, logits).ok()?;
        }

        Some(batch)
    }

    /// The maximum number of members in the batch.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// The current number of members in the batch.
    pub const fn len(&self) -> usize {
        self.batch.n_tokens as usize
    }

    /// Returns true if batch is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The size of each embedding.
    pub const fn embd_len(&self) -> usize {
        self.embd_len
    }

    /// The maximum number of sequence ids per token.
    pub const fn n_seq_max(&self) -> usize {
        self.n_seq_max
    }

    /// The tokens in this batch.
    ///
    /// This will return `None` if the [`Batch`] was created with `embd_len` set
    /// to zero.
    pub fn tokens(&self) -> Option<&[llama_token]> {
        if self.batch.token.is_null() {
            debug_assert!(!self.batch.embd.is_null());
            None
        } else {
            Some(
                &unsafe {
                    std::slice::from_raw_parts(
                        self.batch.token,
                        self.capacity(),
                    )
                }[..self.len() as usize],
            )
        }
    }

    /// The tokens in this batch.
    ///
    /// This will return `None` if the [`Batch`] was created with `embd_len` set
    /// to zero.
    pub fn tokens_mut(&mut self) -> Option<&mut [llama_token]> {
        if self.batch.token.is_null() {
            debug_assert!(!self.batch.embd.is_null());
            None
        } else {
            Some(
                &mut unsafe {
                    std::slice::from_raw_parts_mut(
                        self.batch.token,
                        self.capacity(),
                    )
                }[..self.len() as usize],
            )
        }
    }

    /// The embeddings in this batch at index `i`.
    ///
    /// This will return None if the index is invalid or if the batch was
    /// created with `embd_len` set to zero.
    pub fn embd(&self, i: usize) -> Option<&[f32]> {
        if self.batch.embd.is_null() {
            debug_assert!(!self.batch.token.is_null());
            None
        } else {
            if (i as usize) >= self.len() {
                None
            } else {
                Some(unsafe {
                    std::slice::from_raw_parts(
                        self.batch.embd.add(i * self.embd_len()),
                        self.embd_len(),
                    )
                })
            }
        }
    }

    /// The embeddings in this batch at index `i`.
    ///
    /// This will return None if the index is invalid or if the batch was
    /// created with `embd_len` set to zero.
    pub fn embd_mut(&mut self, i: usize) -> Option<&mut [f32]> {
        if self.batch.embd.is_null() {
            debug_assert!(!self.batch.token.is_null());
            None
        } else {
            if (i as usize) >= self.len() {
                None
            } else {
                Some(unsafe {
                    std::slice::from_raw_parts_mut(
                        self.batch.embd.add(i * self.embd_len()),
                        self.embd_len(),
                    )
                })
            }
        }
    }

    /// The position of a given index in the batch.
    pub const fn pos(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.batch.pos, self.len()) }
    }

    /// The position of a given index in the batch.
    pub fn pos_mut(&mut self) -> &mut [i32] {
        unsafe { std::slice::from_raw_parts_mut(self.batch.pos, self.len()) }
    }

    /// The number of sequence ids for a given index in the batch.
    pub const fn n_seq(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.batch.n_seq_id, self.len()) }
    }

    /// The number of sequence ids for a given index in the batch.
    fn n_seq_mut(&mut self) -> &mut [i32] {
        unsafe {
            std::slice::from_raw_parts_mut(self.batch.n_seq_id, self.len())
        }
    }

    /// Whether logits should be calculated at a given index in the batch.
    pub fn logits(&self) -> &[bool] {
        // Safety: This and the accessor below are safe because we know a bool
        // is the same size as an i8 and we know the 0 and 1 values correspond
        // to false and true. Otherwise the following would not compile:
        static_assertions::assert_eq_size!(bool, i8);
        static_assertions::const_assert_eq!(false as i8, 0);
        static_assertions::const_assert_eq!(true as i8, 1);

        unsafe {
            std::slice::from_raw_parts(
                self.batch.logits as *const bool,
                self.len(),
            )
        }
    }

    /// Whether logits should be calculated at a given index in the batch.
    fn logits_mut(&mut self) -> &mut [bool] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.batch.logits as *mut bool,
                self.len(),
            )
        }
    }

    /// Clear the batch.
    pub fn clear(&mut self) {
        self.batch.n_tokens = 0;
    }

    /// Add a token to the batch.
    pub fn add_token(
        &mut self,
        token: llama_token,
        pos: usize,
        seq_ids: Option<&[llama_seq_id]>,
        logits: bool,
    ) -> Result<(), AddError> {
        let i = self.len();

        if pos >= self.capacity() {
            return Err(AddError::InvalidPosition);
        }

        if i >= self.capacity() {
            return Err(AddError::Full);
        }

        if self.embd_len() != 0 {
            return Err(AddError::ExpectedEmbedding);
        }

        self.batch.n_tokens += 1;

        self.tokens_mut().unwrap()[i] = token;
        self.pos_mut()[i] = pos as i32;

        let sequences = unsafe {
            std::slice::from_raw_parts_mut(self.batch.seq_id, self.len())
        };
        let sequence = unsafe {
            std::slice::from_raw_parts_mut(sequences[i], self.n_seq_max())
        };

        match seq_ids {
            Some(seq_ids) => {
                if seq_ids.len() > self.n_seq_max() {
                    self.batch.n_tokens -= 1;
                    return Err(AddError::InvalidSequenceLength);
                }

                // We want to panic if the number of sequence ids is greater
                // than i32::MAX
                self.n_seq_mut()[i] = seq_ids.len().try_into().unwrap();

                // Safety: This is safe because we control construction of the
                // batch and we know that the sequence ids are valid for the
                // lifetime of the batch. We also know that len is valid because
                // the only way it changes is through our accessor methods.
                sequence[..seq_ids.len()].copy_from_slice(seq_ids);
                sequence[seq_ids.len()..].fill(0);
            }
            None => {
                // There is always at least one sequence id
                self.n_seq_mut()[i] = 1;
                sequence[0] = 0;
            }
        }
        self.logits_mut()[i] = logits;

        Ok(())
    }

    /// Add tokens to the batch.
    pub fn add_tokens<I>(
        &mut self,
        tokens: I,
        pos: usize,
        seq_ids: Option<&[llama_seq_id]>,
        logits: bool,
    ) -> Result<(), AddError>
    where
        I: IntoIterator<Item = llama_token>,
    {
        for token in tokens {
            self.add_token(token, pos, seq_ids, logits)?;
        }

        Ok(())
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.batch) };
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_batch() {
        for n_seq_max in 1..16usize {
            let mut batch = Batch::new(16, 0, n_seq_max).unwrap();

            for i in 0..16 {
                assert_eq!(batch.capacity(), 16);
                assert_eq!(batch.len(), i);
                assert_eq!(batch.embd_len(), 0);
                assert_eq!(batch.n_seq_max(), n_seq_max as usize);
                assert!(batch.tokens().is_some());
                assert!(batch.tokens_mut().is_some());
                assert!(batch.embd(i).is_none());
                assert!(batch.embd_mut(i).is_none());
                assert_eq!(
                    batch.add_token(
                        i as llama_token,
                        i,
                        Some(&vec![42; n_seq_max as usize]),
                        true
                    ),
                    Ok(())
                );
                assert_eq!(batch.n_seq()[i], n_seq_max as i32);
                assert_eq!(batch.logits()[i], true);
                assert_eq!(batch.pos()[i], i as i32);
            }

            batch.clear();

            for i in 0..16_usize {
                assert_eq!(batch.capacity(), 16);
                assert_eq!(batch.len(), i);
                assert_eq!(batch.embd_len(), 0);
                assert_eq!(batch.n_seq_max(), n_seq_max);
                assert!(batch.tokens().is_some());
                assert!(batch.tokens_mut().is_some());
                assert!(batch.embd(i).is_none());
                assert!(batch.embd_mut(i).is_none());
                assert_eq!(
                    batch.add_token(i as llama_token, i, None, false),
                    Ok(())
                );
                assert_eq!(batch.n_seq()[i], 1);
                assert_eq!(batch.logits()[i], false);
                assert_eq!(batch.pos()[i], i as i32);
            }

            // The batch is full
            assert_eq!(
                batch.add_token(16, 15, None, true),
                Err(AddError::Full)
            );
            // The position is invalid
            assert_eq!(
                batch.add_token(16, 16, None, true),
                Err(AddError::InvalidPosition)
            );
        }
    }
}
