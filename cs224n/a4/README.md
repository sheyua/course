# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository

# 1
g. masks zeros out the attention weight on padded words. the attention vector a_t would only be aggregated over the legit words on the original source sentence. those encoded hidden states would be zero (using pad-pack) but might still take softmax weight if their attention is non-zero, otherwise the a_t would be biased to zero for short sentences.
i. i) dot use less parameters, dot enforce the encoder and decoder to be on the same dimension
   ii) additive allows non-linearity in the attention, however it is less intuitive without the direction interaction between encoded h and decoded s.


# 2
a. there would be too many words in Cherokee given its polysynthetic nature, SentencePiece tokenize based on unsupervised training of subword set and give manageable vocabulary.
b. they find the building blocks of the words rather than using the words alone. for synthetic words, it could break them down to subwords.
c. the similarity between languages allow the model to transfer some of the weights update based on another language to the low-resource language. in the stats way of looking at this, it shrinks the low-resource language model to the multilingual average. the intuitive idea behind it is that there is some "universal understanding" that acts as the intermediate step of translation, somewhere in the neural network layers.
d. i) the structure of "crown of daisies" might be less observed and does not resonate with the decoder. could be solved by increasing model complexity or with more data.
   ii) "Ulihelisdi" has a more or less direct translation to joy, so may the expression of "nigalisda" is not so obviously related with she. could be solved by increasing model complexity or with more data.
   iii) "usdi atsahi" is possibly small fish, the issue is not about small fish it is about that the term "yitsadawoesdi" does not get translated which could mean swin. could be solved by increasing model complexity or with more data.
e. i) translation has:  
         [therefore the chief priests and the officers]1, cried with him, [saying, crucify him, crucify him!]2 for I am not with him: for I am not in him: for I am not in him: for I am not with him. For I am not in him. For I am not in him. For I have no longer in him.
      target has:
         When [therefore the chief priests and the officers]1 saw him, they cried out, [saying, Crucify him, crucify him!]2 Pilate saith unto them, Take him yourselves, and crucify him: for I find no crime in him.
      the training data does not have the exact phrases, however it has the sub-phrases of 'the chief priests' in phrase 1 and it has similar phrases like phrase 2, e.g. 'saying, Crucify, crucify him'.
   ii) still using the previous example, the translation went off keep repeating "I am not in him". This is a commonly observed pattern that the translation does not know how to stop. Maybe it should also pay attention what it has said.
f. i) 1-gram:
	c1: the [love] [can] [always] do, total = 5
	c2: [love] [can] make [anything] [possible], total = 5
	p1 of c1 = 3/5
	p1 of c2 = 4/5
      2-gram:
	c1 the-love [love-can] [can-always] always-do, total = 4
	c2 [love-can] can-make make-anything [anything-possible], total = 4
	p2 of c1 = 2/4
	p2 of c2 = 2/4
      BP:
	c1 len(c) = 5, len(r) = 4, BP = 1
	c2 len(c) = 5, len(r) = 4, BP = 1
      BLEU:
	c1 = exp(.5 * log(3/5) + .5 * log(2/4)) = sqrt(3/10)
	c2 = exp(.5 * log(4/5) + .5 * log(2/4)) = sqrt(4/10)
    c2 is a better translation and I agree.
   ii) 1-gram:
	c1: the [love] [can] [always] do, total = 5 
	c2: [love] [can] make anything possible, total = 5
	p1 of c1 = 3/5
	p1 of c2 = 2/5
      2-gram:
	c1 the-love [love-can] [can-always] always-do, total = 4
	c2 [love-can] can-make make-anything anything-possible, total = 4
	p2 of c1 = 2/4
	p2 of c2 = 1/4
      BP:
	c1 len(c) = 5, len(r) = 6, BP = exp(-1/5)
	c2 len(c) = 5, len(r) = 6, BP = exp(-1/5)
      BLEU:
	c1 = exp(.5 * log(3/5) + .5 * log(2/4)) = sqrt(3/10) * exp(-1/5)
	c2 = exp(.5 * log(2/5) + .5 * log(1/4)) = sqrt(1/10) * exp(-1/5)
      c1 now becomes a better translation, I do not agree c2 is still closer to r1 meaning-wise.
   iii) it is a problem because the single reference can be rewritten completely differently to mean the same thing. also some part of the sentence could contain more meaningful things per unit word. if the single reference does not have bias then evaluating on large corpus could solve the problem, though resulting in a lower score on average.
   iv) systematic avoiding human bias and automatic can be applied on large scale and compare things across the borad.
       could be problematic due to data availability which takes time and efforts to collect and the treatment is too mechanical both on the n-gram method and length.

