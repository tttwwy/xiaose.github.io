---
title: Theano调试技巧
tags:
  - 原创
  - 深度学习
abbrlink: 15149
date: 2017-01-11 10:00:00
---
Theano是最老牌的深度学习库之一。它灵活的特点使其非常适合学术研究和快速实验，但是它难以调试的问题也遭到过无数吐槽。其实Theano本身提供了很多辅助调试的手段，下面就介绍一些Theano的调试技巧，让Theano调试不再难。而关于深度学习的通用调试技巧，请参见我之前的文章：{% post_link 深度学习调参技巧 [深度学习调参技巧] %}。 
> 以下的技巧和代码均在Theano 0.8.2 上测试通过，不保证在更低的版本上也可以适用。

# 如何定位出错位置
Theano的网络在出错的时候，往往会提供一些出错信息。但是出错信息往往非常模糊，让人难以直接看出具体是哪一行代码出现了问题。大家看下面的例子：<!-- more -->
```python
import theano
import theano.tensor as T
import numpy as np
x = T.vector()
y = T.vector()
z = x + x
z = z + y
f = theano.function([x, y], z)
f(
np.array([1,2],dtype=theano.config.floatX), np.array([3,4,5],dtype=theano.config.floatX))
```
将代码保存到test.py文件中，在命令行中执行：
```bash
THEANO_FLAGS="device=gpu0,floatX=float32" python test.py
```
输出结果如下：
```python
Traceback (most recent call last):
  File "test.py", line 10, in <module>    print f(np.array([1,2],dtype='float32'), np.array([3,4,5],dtype='float32'))
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/function_module.py", line 871, in __call__
    storage_map=getattr(self.fn, 'storage_map', None))  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/link.py", line 314, in raise_with_op    reraise(exc_type, exc_value, exc_trace)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/function_module.py", line 859, in __call__
    outputs = self.fn()
ValueError: GpuElemwise. Input dimension mis-match. Input 1 (indices start at 0) has shape[0] == 2, but the output size on that axis is 3.
Apply node that caused the error: GpuElemwise{Composite{((i0 + i1) + i0)}}[(0, 0)](GpuFromHost.0, GpuFromHost.0)
Toposort index: 2
Inputs types: [CudaNdarrayType(float32, vector), CudaNdarrayType(float32, vector)]
Inputs shapes: [(3,), (2,)]
Inputs strides: [(1,), (1,)]
Inputs values: [CudaNdarray([ 3.  4.  5.]), CudaNdarray([ 1.  2.])]
Outputs clients: [[HostFromGpu(GpuElemwise{Composite{((i0 + i1) + i0)}}[(0, 0)].0)]]

HINT: Re-running with most Theano optimization disabled could give you a back-trace of when this node was created. This can be done with by setting the Theano flag 'optimizer=fast_compile'. If that does not work, Theano optimizations can be disabled with 'optimizer=None'.
HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.
```
比较有用的信息是：Input dimension mis-match，但是具体出问题在哪里，仍然让人一头雾水。因为Theano的计算图进行了一些优化，导致出错的时候难以与原始代码对应起来。想解决这个也很简单，就是关闭计算图的优化功能。可以通过THEANO_FLAGS的optimizer,它的默认值是"fast_run"，代表最大程度的优化，我们平时一般就使用这个，但是如果想让调试信息更详细，我们就需要关闭一部分优化:fast_compile或者关闭全部优化：None，这里我们将optimizer设置成"None"，执行如下命令：
```bash
THEANO_FLAGS="device=gpu0,floatX=float32,optimizer=None" python test.py
```
结果如下：
```python
Traceback (most recent call last):
  File "test.py", line 10, in <module>
    print f(np.array([1,2],dtype='float32'), np.array([3,4,5],dtype='float32'))
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/function_module.py", line 871, in __call__
    storage_map=getattr(self.fn, 'storage_map', None))
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/link.py", line 314, in raise_with_op
    reraise(exc_type, exc_value, exc_trace)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/function_module.py", line 859, in __call__
    outputs = self.fn()
ValueError: Input dimension mis-match. (input[0].shape[0] = 3, input[1].shape[0] = 2)
Apply node that caused the error: Elemwise{add,no_inplace}(<TensorType(float32, vector)>, <TensorType(float32, vector)>)
Toposort index: 0
Inputs types: [TensorType(float32, vector), TensorType(float32, vector)]
Inputs shapes: [(3,), (2,)]
Inputs strides: [(4,), (4,)]
Inputs values: [array([ 3.,  4.,  5.], dtype=float32), array([ 1.,  2.], dtype=float32)]
Outputs clients: [[Elemwise{add,no_inplace}(Elemwise{add,no_inplace}.0, <TensorType(float32, vector)>)]]

Backtrace when the node is created(use Theano flag traceback.limit=N to make it longer):
  File "test.py", line 7, in <module>
    z = y + x

HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.

```
可以看到，这次直接提示出错的位置在代码的第7行：z = y + x，这个是不是方便很多了呢？

# 如何打印中间结果
下面分别介绍Test Values和Print两种方法。
## 使用Test Values
我曾见过有人为了保证中间运算的实现没有问题，先用numpy实现了一遍，检查每一步运算结果符合预期以后，再移值改成Theano版的，其实大可不必这么折腾。Theano在0.4.0以后，加入了test values机制，简单来说，就是在计算图编译之前，我们可以给symbolic提供一个具体的值，即test_value，这样Theano就可以将这些数据，代入到symbolic表达式的计算过程中，从而完成计算过程的验证，并可以打印出中间过程的运算结果。
大家看下面的例子：
```python
import theano
import theano.tensor as T
import numpy as np
x = T.vector()
x.tag.test_value = np.array([1,2],dtype=theano.config.floatX)

y = T.vector()                                      y.tag.test_value = np.array([3,4,5],dtype=theano.config.floatX)
z = x + x
print z.tag.test_value
z = z + y
print z.tag.test_value
f = theano.function([x, y], z)
```
运行的时候，需要注意，如果需要使用test_value,那么需要设置一下compute_test_value的标记，有以下几种
- off: 关闭，建议在调试没有问题以后，使用off，以提高程序速度。
- ignore: test_value计算出错，不会报错
- warn: test_value计算出错，进行警告
- raise: test_value计算出错，会产出错误
- pdb: test_value计算出错，会进入pdb调试。pdb是python自带的调试工具，在pdb里面可以单步查看各变量的值，甚至执行任意python代码，非常强大，如果想看中间过程，又懒得打太多print，那么可以import pdb
然后在你想设断点的地方加上：pdb.set_trace()，后面可以用指令n单步，c继续执行。更详细的介绍可以参考这里：https://docs.python.org/2/library/pdb.html

下面继续回到test_value，我们将test_value值修改成warn，执行：
```bash
THEANO_FLAGS="device=gpu0,floatX=float32,compute_test_value=warn" python test.py
```
结果如下：
```python
[ 2.  4.]
log_thunk_trace: There was a problem executing an Op.
Traceback (most recent call last):
  File "test.py", line 12, in <module>
    z = z + y
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/tensor/var.py", line 135, in __add__
    return theano.tensor.basic.add(self, other)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/op.py", line 668, in __call__
    required = thunk()
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/op.py", line 883, in rval
    fill_storage()
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/cc.py", line 1707, in __call__
    reraise(exc_type, exc_value, exc_trace)
  File "<string>", line 2, in reraiseValueError: Input dimension mis-match. (input[0].shape[0] = 2, input[1].shape[0] = 3)

```
可以看到，第一个z的值[2,4]被print了出来，同时在test_value的帮助下，错误信息还告诉我们在执行z = z + y 这一行的时候有问题。因此test_value也可以起到，检测哪一行出错的功能。
**小技巧: **人工一个个构造test_value,实在太麻烦，因此可以考虑在训练开始前，从训练数据中随机选一条，作为test_value,这样还能辅助检测，训练数据有没有问题。

## 使用Print
不过test_value对scan支持的不好，而如果网络包含RNN的话，scan一般是不可或缺的。那么如何打印出scan在循环过程中的中间结果呢？这里我们可以使用
theano.printing.Print()，代码如下：
```python
import theano
import theano.tensor as T
import numpy as np
x = T.vector()
y = T.vector()
z = x + x
z = theano.printing.Print('z1')(z)
z = z + y
z = theano.printing.Print('z2')(z)
f = theano.function([x, y], z)
f(np.array([1,2],dtype=theano.config.floatX),np.array([1,2],dtype=theano.config.floatX))
```
执行：
```bash
THEANO_FLAGS="device=gpu0,floatX=float32" python test.py
```
结果如下：
```python
z1 __str__ = [ 2.  4.]
z2 __str__ = [ 3.  6.]
```
不过下面有几点需要注意一下：
- 因为theano是基于计算图的，因此各变量在计算图中被调用执行的顺序，不一定和原代码的顺序一样，因此变量Print出来的顺序也是无法保证的。
- Print方法，会比较严重的拖慢训练的速度，因此最终用于训练的代码，最好把Print去除。
- Print方法会阻止一些计算图的优化，包括一些结果稳定性的优化，因此如果程序出现Nan问题，可以考虑把Print去除，再看看。

# 如何处理Nan
Nan是我们经常遇到的一个问题，我之前的文章：[深度学习网络调试技巧](https://zhuanlan.zhihu.com/p/20792837?refer=easyml) 提到了如何处理Nan问题，其中最重要的步骤，是确定Nan最开始出现的位置。
一个比较暴力的方法，是打印出变量的中间结果，看看Nan是从哪里开始的，不过这样工作量有点太大了。所以这里介绍另外一个比较省事的方法：NanGuardMode。NanGuardMode会监测指定的function，是否在计算过程中出现nan,inf。如果出现，会立刻报错，这时配合前面提到的optimizer=None，我们就可以直接定位到，具体是哪一行代码最先出现了Nan问题。代码如下：
```python
import theano
import theano.tensor as T
import numpy as np
from theano.compile.nanguardmode import NanGuardMode
x = T.matrix()
w = theano.shared(np.random.randn(5, 7).astype(theano.config.floatX))
y = T.dot(x, w)
fun = theano.function(
            [x], y,
mode=NanGuardMode(nan_is_error=True,inf_is_error=True, big_is_error=True)
)
infa = np.tile(
            (np.asarray(100.) ** 1000000).astype(theano.config.floatX), (3, 5))
fun(infa)
```
执行：
```bash
THEANO_FLAGS="device=gpu0,floatX=float32,optimizer=None" python test.py
```

结果如下：
```python
Traceback (most recent call last):
  File "test.py", line 12, in <module>
    f(np.array([1,2],dtype='float32'),np.array([0,0],dtype='float32'))
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/function_module.py", line 859, in __call__
    outputs = self.fn()
Backtrace when the node is created(use Theano flag traceback.limit=N to make it longer):
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/link.py", line 1014, in f
    raise_with_op(node, *thunks)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/link.py", line 314, in raise_with_op
    reraise(exc_type, exc_value, exc_trace)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/link.py", line 1012, in f
    wrapper(i, node, *thunks)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/nanguardmode.py", line 307, in nan_check
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/gof/link.py", line 1012, in f
    wrapper(i, node, *thunks)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/nanguardmode.py", line 302, in nan_check
    do_check_on(x[0], node, fn, True)
  File "/home/wangzhe/anaconda2/lib/python2.7/site-packages/theano/compile/nanguardmode.py", line 272, in do_check_on
    raise AssertionError(msg)
AssertionError: Inf detected
Big value detected
NanGuardMode found an error in an input of this node.
Node:
dot(<TensorType(float32, matrix)>, HostFromGpu.0)
The input variable that cause problem:
dot [id A] ''   
 |<TensorType(float32, matrix)> [id B]
 |HostFromGpu [id C] ''   
   |<CudaNdarrayType(float32, matrix)> [id D]


Apply node that caused the error: dot(<TensorType(float32, matrix)>, HostFromGpu.0)
Toposort index: 1
Inputs types: [TensorType(float32, matrix), TensorType(float32, matrix)]
Inputs shapes: [(3, 5), (5, 7)]
Inputs strides: [(20, 4), (28, 4)]
Inputs values: ['not shown', 'not shown']
Outputs clients: [['output']]

Backtrace when the node is created(use Theano flag traceback.limit=N to make it longer):
  File "test.py", line 8, in <module>
    y = T.dot(x, w)

HINT: Use the Theano flag 'exception_verbosity=high' for a debugprint and storage map footprint of this apply node.

```
可以看到，是y = T.dot(x, w)这一行，产生的Nan.

# 其他
上面的几个技巧，相信可以解决大部分Theano调试中遇到的问题. 同时我们在用Theano实现一些网络结构，例如LSTM的时候，除了直接参考论文之外，这里强烈推荐参考keras进行实现。keras是一个更顶层的库，同时支持Theano和Tensorflow作为后台，里面大部分模型的实现都很可靠，可以学习和参考。
# 参考资料
- http://deeplearning.net/software/theano/library/compile/nanguardmode.html#nanguardmode
- http://deeplearning.net/software/theano/tutorial/debug_faq.html