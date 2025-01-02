# Topic Modeling in Chinese

Topic modeling in Chinese is a substantially different endeavour from doing it in Indo-European languages, such as English.
To offset for the complexity introduced by this, we include a submodule in Turftopic for Chinese topic modeling.

There are two steps in the topic modeling pipeline that need to be altered in order for models to be usable with Chinese.

## Tokenization

Turftopic uses the [jieba](https://github.com/fxsjy/jieba) tokenizer for tokenizing Chinese text.
We include a Chinese version of `CountVectorizer` that uses Jieba by default, and has an optionally applicable Chinese stop word list.

```python
from turftopic.chinese import ChineseCountVectorizer

vectorizer = ChineseCountVectorizer(min_df=10, stop_words="chinese")
```

We also provide a sensible default which is just a `ChineseCountVectorizer` with `min_df=10` and `stop_words="chinese"`:

```python
from turftopic.chinese import default_chinese_vectorizer

vectorizer = default_chinese_vectorizer()
```

## Encoder

You will need to use a different encoder model for your topic models, since `all-MiniLM-L6-v2` does not support Chinese by default.
We recommend the [BGE-zh model family](https://huggingface.co/collections/BAAI/bge-66797a74476eb1f085c7446d) for this purpose.

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
```

## Defining a Chinese topic model

Once having defined these steps, you can pass these arguments when initializing your topic model.

```python
from turftopic import KeyNMF

model = KeyNMF(
    n_components=20,
    encoder=SentenceTransformer("BAAI/bge-small-zh-v1.5"),
    vectorizer=default_chinese_vectorizer(),
    random_state=42,
)
model.fit(corpus)

model.print_topics()
```

| Topic ID | Highest Ranking |
| - | - |
| 0 | 消息, 时间, 科技, 媒体报道, 美国, 据, 国外, 讯, 宣布, 称 |
| 1 | 体育讯, 新浪, 球员, 球队, 赛季, 火箭, nba, 已经, 主场, 时间 |
| 2 | 记者, 本报讯, 昨日, 获悉, 新华网, 基金, 通讯员, 采访, 男子, 昨天 |
| 3 | 股, 下跌, 上涨, 震荡, 板块, 大盘, 股指, 涨幅, 沪, 反弹 |
| 4 | 像素, 相机, 佳能, 镜头, 数码相机, 单反, 报价, 价格, 单反相机, 尼康 |
| 5 | 股票, 该股, 投资者, 分析, 新浪, 个人观点, 以沪深, 交易所, 注意, 新闻报道 |
| 6 | 新浪, 娱乐, 讯, 出席, 近日, 香港, 亮相, 微博, 举行, 拍摄 |
| 7 | 户型, 样板间, 房产, 均价, 项目, 平米, 入住, 位于, 地图搜索, 论坛 |
| 8 | 市场, 经济, 预期, 国内, 资金, 行业, 投资, 价格, 品牌, 影响 |
| 9 | 北京, 时间, 凌晨, 新浪, 8, 11, 科技, 晚间, 美元, 10 |
| 10 | 将, 宣布, 举行, 表示, 正式, 与, 进行, 预计, 可能, 称 |
| 11 | 报道, 据, 香港, 香港媒体, 日电, 中新网, 媒体, 本报记者, 英国, 国际 |
| 12 | 中国, 发展, 国际, 全球, 2010, 举行, 企业, 发布, 国内, 互联网 |
| 13 | 公司, 投资, 基金, 股份, 亿元, 宣布, 收购, 该, 上市, 证券 |
| 14 | 考生, 考试, 高考, 招生, 公布, 今年, 录取, 教育, 2010, 成绩 |
| 15 | 财经, 讯, 新浪, 盈利, 消息, 评级, 预期, 业绩, 增长, 港元 |
| 16 | 对, 一个, 新, 已经, 表示, 成为, 问题, 进行, 还, 会 |
| 17 | 比赛, 主场, 体育讯, 结束, 一场, 对手, 联赛, 最后, 中国队, 冠军 |
| 18 | 3, 10, 4, 5, 日电, 6, 2, 后, 发生, 7 |
| 19 | 目前, 元, 笔记本, 产品, 一款, 售价, 液晶电视, 价格, 处理器, 推出 |

