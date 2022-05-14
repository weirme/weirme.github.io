---
title: Python Mathematic Modules
date: 2018-08-20
categories: [Notes, Python]
tags: [Python, Numpy, Matplotlib, Pandas]
math: true
---

## numpy

### Import

It is common to import `numpy`  under a briefer name `np` and also import some class frequently used like `array`. Common import as follows:

```python
import numpy as np
from np import array
```

### Array

#### Ways to create and some properties

We use `array()` function to generate an array object with a list as parameter. (The list can be multdimensional with nested `[]` operator.)

The `shape` property of an array returns a tuple with the size of each array dimension.

Sometimes we use `np.random.randint()` function to generate a random array. 3 arguments are required. First two represent range of random integer, the third arguments `size=` represents dimensionality(`shape`) of this array. As follows:

```python
a = np.random.randint(1, 8, size=(2, 3, 4))
```

Besides, `linspace()` function can also generate an array. First two arguments of it is start and stop of a interval, at most cases we pass an integer representing numbers of elements in the array to be generated,  and many other arguments are optional. If we let `endpoint=True,` then endpoints of the interval must be included in the new array.

When used with an array, the `len()` function returns the length of the first axis.

#### Methods to modify

Arrays can be reshaped passing tuple that specify new dimensions as parameter to `reshape()` function. (Only one parentheses is also permitted.)  Following code generates a  $4\times 1$ row vector.

```python
a = array([1, 2, 3, 4])
a.reshape((2, 1))
```

Notice that the `reshape()` function **creates a new array** rather than modifying the original one.

One-dimensional versions of multi-dimensional arrays can be generated with `flatten()` function.

Two or more arrays can be concatenated together using the `concatenate()` function with a tuple of the arrays to be joined. We can pass the second parameter represented a specified axis to which the arrays concatenate.

Dimensionality of an array can be increased using `newaxis` constant in bracket notation as follows:

```python
a = array([1, 2, 3, 4])
b = a[np.newaxis,:]
c = a[:,np.newaxis]
```

`b.shape` output is `(1, 2)` while `c.shape` output is `(2, 1)`.  It's obvious that when `newaxis` is in front,  a dimensionality added to front and when it is in the rear, a dimensionality added to rear. Similarly, if $a\in\mathbb{R}^{2\times 3\times 4} $, then we can use `newaxis` like `a[:, np.newaxis, :, :]`, and a new dimensionality added to the corresponding position.

#### Iteration

It is possible to iterate over arrays in a manner similar to that of lists using `for` syntax.But for multidimensional arrays, iteration proceeds over the first axis such that each loop returns a subsection of the array. In such case, multiple assignment can also be used with array iteration as follows:

```python
a = np.array([[1, 2], [3, 4], [5, 6]])
for (x, y) in a:
    print x * y
```

#### Operation

Array object owns many functions to implement basic array operations.

|  Function  |    Description     |  Function  |     Description     |
| :--------: | :----------------: | :--------: | :-----------------: |
|  `sum()`   |        sum         |  `prod()`  |       product       |
|  `mean()`  |      average       |  `var()`   |      variance       |
|  `std()`   | standard deviation | `median()` |       median        |
|  `min()`   |      minimum       |  `max()`   |       maximum       |
| `argmin()` | indices of minimum | `argmax()` | indices of maximum  |
|  `sort()`  |        sort        | `unique()` | get unique elements |

For multidimensional arrays, each of the functions thus far described can take an optional argument `axis` that will perform an operation along only the specified axis, placing the results in a return array

There are many operations prepared for vactor and matrix mathematics in `numpy`. Add prefix `numpy.` to call them.

|     Function     |         Description          |        Function        |         Description          |
| :--------------: | :--------------------------: | :--------------------: | :--------------------------: |
|    `inner()`     |        inner product         |       `outer()`        |        outer product         |
|     `dot()`      |        matrix product        |     `tensordot()`      |        tensor product        |
|   `diagonal()`   |    get diagonal elements     |     `transpose()`      |          transpose           |
|  `linalg.det()`  |         determinant          |     `linalg.inv()`     |           inverse            |
| `linalg.pinv()`  |        pseudo-inverse        |    `linalg.norm()`     |             norm             |
|  `linalg.eig()`  | eigenvalues and eigenvectors | `linalg.matrix_rank()` |             rank             |
| `linalg.solve()` |    solve matrix equation     |     `linalg.svd()`     | singular value decomposition |


## matplotlib

### Some supplementary explanation to pyplot module

#### Make a straight line on the vertical $x$ axis

This is a special syntax as follows:

```python
plot([x, x], [0, y])
```

#### Add subplot

We call `subplot()` function to add a subplot to the figure. This function needs 3 integers $a, b, c$ as arguments, which means dividing figure to $a\times b$ blocks, and the new plot will be added to the $c$-th block.

To add several subplot, we can call `subplots()` function which return an array of each subplot. Pass the first and second argument as the row and column of the array respectively, and the third is usually keyword `figsize=`.


#### Setting ticks and labels

In `x/yticks()` function, we pass an array generated by `np.linspace()` as the first arguments. It's worth mentioning that an list of str can perform as the second argument representing label of corresponding tick. LateX string is also permitted to be passed in a raw string.

```python
xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],[r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
```

#### Moving spines

We can discard the top and right spines by setting their color to none and move the bottom and left ones to coordinate 0 in data space coordinates with code as follows:

```python
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
```

#### Adding a legend

Adding a keyword argument `label=` to `plot()` function, and using `legend()` function with `loc=` as argument to add a legend to the specific plot.


### Use `hist()` function to generate a histogram

Firstly, we pass an array as argument representing dataset. And some other optional keyword arguments are listed below:

|  Argument  |                         Description                          |
| :--------: | :----------------------------------------------------------: |
|  `bins=`   |                      set number of bins                      |
| `density=` | If `True`, the first element of the return tuple will be the counts normalized to form a probability density |
|  `color=`  |                      set color of bins                       |
|  `alpha=`  |                       set transparency                       |


## pandas

ATTENTION `Pandas` ! NOT  `panda` !

### Read data

Here are some functions in `pandas` that are uesd to read data from different kind of file.

|      Function      |            Description            |
| :----------------: | :-------------------------------: |
|    `read_csv()`    |           read CSV file           |
|   `read_json()`    |          read JSON file           |
|   `read_excel()`   |          read Excel file          |
|    `read_sql()`    |         read SQL database         |
|   `read_html()`    |   read URL, string or html file   |
| `read_clipboard()` |     read data from clipboard      |
|   `Dataframe()`    | import data from Dataframe object |


### Write data

Here are some functions in `pandas` that are uesd to write data into different kind of file.

|   Function   |       Description       |
| :----------: | :---------------------: |
|  `to_csv()`  |   write into CSV file   |
| `to_josn()`  |  write into JSON file   |
| `to_excel()` |  write into Excel file  |
|  `to_sql()`  | write into SQL database |


### Dataframe and Series

Dataframe is the most important object of `pandas` which used to handle a large variety of  files. We will discuss this object next.

`count()` function returns number of samples in Dataframe.

`[]` operator included name of one column can be used to get all elements in this column, and these elements are returned as a Series object, which is another important object of `pandas`.

`[]` operator can include a bool expression as well. In code `df[df['class'] == 'Iris-setosa']`, we get all samples returned as a new Dataframe whose property `class` is equal to `'Iris-setosa'`.

Another way to split data is using Dataframe object property `ix` with row and column index as arguments. `[:]` syntax is also available in this function, but when using this syntax, notice follow situation: code `ix[:3, :2]` gets datas in **first 4 rows** and first 2 columns.

Further more, Python list iterator is also availble in `ix` property.

Series object is a column in Dataframe, which usually used to describe one of prosperties of the sample. We can use `unique()` function to delete all repeat values in a single column and get unique values remaining returned as an array. As follows:

```python
df['class'].unique()
```

After getting the new Dataframe, we can use `reset_index(drop=True)` function to reset data index from 0, which makes it easier for further processing on these data.

We call `describe()` function of Dataframe to get statistic data of the specific dataset. Output is similar as below:

```python
       sepal length  sepal width  petal length  petal width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
```

In addition, call `corr()` function to show the correspondence between properties of samples in dataset.
