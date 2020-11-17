# Embedding层封装
## word2vec
### 创建embedder
```python
from word2vec import word2vec_Embedder
word2vec_embedder = word2vec_Embedder(word_file='./data/bert_vocab.txt',
                                      word2vec_file='./data/glove.840B.300d.txt',
                                      static=False,
                                      use_gpu=True,
                                      npy_file=None,
                                      word_dim=None)
```
- `word_file`: 特定任务对应的词库，一行为一个单词。自动添加`@UNKNOW_TOKEN@`和`@PADDING_TOKEN@`。
```
word1
word2
word3
....
```
- `word2vec_file`: 词向量文件。例如glove, 见https://nlp.stanford.edu/projects/word2vec/
- `static`: 表示是否更新word2vec embedding参数
- `use_gpu`: 是否使用gpu
- `npy_file`: 第一次从txt后自动在给定path下创建npy文件，第二次就不用从txt读取了，耗时
- `word_dim`: 如果没有给定word2vec_file和npy_path, 给定word_dim自动初始化参数
### 使用embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding = word2vec_embedder(tokens_list)
```
- `tokens_list:` `batch_size`个切分为单词列表的句子，不用一样长
- `embedding`: 一个`(batch_size x max_length x word_dim)`的Tensor.多余的对应着`@PADDING_TOKEN@`的embedding.
### 其他
如果提供的文件为空文件, 则词向量按正态分布随机初始化, 因此这个模块也可以用来作为`pos_tag`等的embedding.
# Bert
### 创建embedder
```python
from bert import Bert_Embedder
bert_embedder = Bert_Embedder(vocab_dir='./data/bert_vocab.txt',  # bert词表
                              bert_model_dir='/home/wpy/data/word2vec/bert-base-cased', # bert预训练模型，里面有一个json文件和一个bin文件
                              output_all_encoder_layers=False,
                              split=True,
                              use_gpu=True)
```
- `vocab_dir`: bert词表
- `bert_model_dir`: bert预训练模型参数
- `split`: 是否对输入的每一个单词进行再切分。如果不切分，则不存在bert词表中的单词被替换为`[UNK]`。如果切分, 那么单词会被切分为更小单元，
如`trainyou -> train, ##you`, 最后`trainyou`的embedding为`train, ##you`两者的平均
- `use_gpu`: 是否使用gpu训练
### 使用embedder
```python
tokens_list=[['i','hate','this'],['i','am','your','friend']]
embedding, pooled_out = bert_embedder(tokens_list)
```
- 输入:
    - `tokens_list:`同word2vec输入一样, `batch_size`个切分为单词列表的句子，不用一样长.
    - `token_type_ids`: 如果希望不连续地给定token type，可以给定这个参数，如果没给定，模型更具[SEP]进行切割，前面type=0，后面type=1
- 输出:
    - `embedding`: 
        - 若`split`为`True`: `(batch_size x max_length+2 x word_dim)`的Tensor, `[PAD]`的向量为0
        - 若`split`为`False`: 返回`(batch_size x max_length + 2 x word_dim)`的Tensor, , 由于原始bert的实现原因, 多余的`[PAD]`的embedding不是0.
    - `pooled_out`:`Tensor(n, hidden_size`): 每个句子最后一层encoder的第一个词`[CLS]`经过Linear层和激活函数`Tanh()`后的Tensor. 其代表了句子信息
    
