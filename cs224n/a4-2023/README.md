# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_basic_nmt repository

g. masks zeros out the attention weight on padded words. the attention vector a_t would only be aggregated over the legit words on the original source sentence. those encoded hidden states would be zero (using p
ad-pack) but might still take softmax weight if their attention is non-zero, otherwise the a_t would be biased to zero for short sentences.
i. i) dot use less parameters, dot enforce the encoder and decoder to be on the same dimension
   ii) additive allows non-linearity in the attention, however it is less intuitive without the direction interaction between encoded h and decoded s.


