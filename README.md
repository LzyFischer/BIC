# BIC
## BIC: Twitter Bot Detection with Text-Graph Interaction and Semantic Consistency

Twitter bots are automatic programs operated by malicious actors to manipulate public 
opinion and spread misinformation. Research efforts have been made to automatically 
identify bots based on texts and networks on social media. Existing methods only leverage 
texts or networks alone, and while few works explored the shallow combination of the 
two modalities, we hypothesize that the interaction and information exchange between 
texts and graphs could be crucial for holistically evaluating bot activities on social
media. In addition, according to a recent survey (Cresci, 2020), Twitter bots are 
constantly evolving while advanced bots steal genuine users' tweets and dilute their
malicious content to evade detection. This results in greater inconsistency across the
timeline of novel Twitter bots, which warrants more attention. In light of these 
challenges, we propose BIC, a Twitter Bot detection framework with text-graph 
Interaction and semantic Consistency. Specifically, in addition to separately modeling 
the two modalities on social media, BIC employs a text-graph interaction module to
enable information exchange across modalities in the learning process. In addition, 
given the stealing behavior of novel Twitter bots, BIC proposes to model semantic 
consistency in tweets based on attention weights while using it to augment the decision
process. Extensive experiments demonstrate that BIC consistently outperforms 
state-of-the-art baselines on two widely adopted datasets. Further analyses reveal 
that text-graph interactions and modeling semantic consistency are essential 
improvements and help combat bot evolution.


## Preprocessed Data
We provide our preprocessed data:

link: [https://pan.baidu.com/s/1CWT_rkggu3nB2JYn6xny4Q](https://pan.baidu.com/s/1CWT_rkggu3nB2JYn6xny4Q)

password: 6wbt


## Reproduce
For **Twibot-20** dataset, we recommend the code in the Twibot20 folder. For **Cresci-15** dataset, 
we recommend the code in the Cresci15v2 dataset.

**We will give a better illustration of the code sooner!**
