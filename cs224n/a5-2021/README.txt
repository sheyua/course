2.g.ii In a single layer, the synthesizer self-attention might not be able to fully replicate what the KQV self-attention. The KQV self-attention mechanism tends to be more expressive due to its separate transformations for keys, queries, and values. This separation allows the model to learn intricate patterns and relationships within the data. A single-layer synthesizer self-attention may lack the complexity required to capture the same level of detail in the relationships between elements. 

3.a the pretrained model has a better understanding of language by itself due to the learning on wiki text, while the non-pretrained one does not. 
3.b 1) it might mislead human user, making up facts while user could take them as granted, 2) this lack of accuracy and also an obvious way of sanity checking the answer makes downstream task less reliable.
3.c it might look for similar names, last names in particular, in the data it has been trained on and output the most likely birth place. it lacks reliability for out of domain task, it should just have said no-answer.






