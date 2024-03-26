![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document

2024-03-20 parallel-python

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

[link](https://esciencecenter-digital-skills.github.io/2024-03-19-ds-parallel/)

🛠 Setup

[link](https://esciencecenter-digital-skills.github.io/parallel-python-workbench/index.html)


## 👩‍🏫👩‍💻🎓 Instructors

Johan Hidding, Francesco Nattino, Flavio Hafner

## 🧑‍🙋 Helpers

Sander van den Rijn


## 👩‍💻👩‍💼👨‍🔬🧑‍🔬🧑‍🚀🧙‍♂️🔧 Roll Call
Name/ pronouns (optional) / job, role / social media (twitter, github, ...) / background or interests (optional) / city

*snipped*

## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 09:30 | Welcome, icebreaker and recap |
| 09:45 | Threads and processes |
| 10:30 | Coffee break | 
| 10:45 | Delayed evaluation |
| 12:00 | Tea break |
| 12:15 | Data flow patterns |
| 13:00 | Lunch |
| 14:00 | Introduction to coroutines and asyncio |
| 14:45 | Coffee break |
| 15:00 | Computing fractals in parallel |
| 15:45 | Tea break |
| 16:30 | Presentations of group work |
| 16:45 | Post-workshop Survey |
| 17:00 | Drinks |

## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises

:::info
### Challenge: Run the workflow

Given this workflow:
```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, -3)
```
Visualize and compute `y_p` and `z_p` separately, how many times is `x_p` evaluated?

Now change the workflow:

```python
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir="LR")
```

We pass the not-yet-computed promise `x_p`to both `y_p`and `z_p`. Now, only compute `z_p`, how many times do you expect `x_p` to be evaluated? Run the workflow to check your answer.
:::


#### Solution

```python!
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, -3)

z_p.visualize()
z_p.compute()
# we see that x_p is computed twice
```

The example illustrates that computations are re-used across tasks and data can be reused as much as possible. The following computes `x_p` only once.

```python!
x_p = add(1, 2)
y_p = add(x_p, 3)
z_p = add(x_p, y_p)
z_p.visualize(rankdir="LR")
```

---



:::info
### Challenge: Design a mean function and calculate $\pi$

Write a delayed function that computes the mean of its arguments. Use it to esimates $\pi$ several times and returns the mean of the results.

```python
>>> mean(1, 2, 3, 4).compute()
2.5
```
Make sure that the entire computation is contained in a single promise.

Here is the function to estimate $\pi$ from N random samples:
```python
import random

def calc_pi(N):
    """Computes the value of pi using N random samples."""
    M = 0
    for i in range(N):
        # take a sample
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y < 1.: M+=1
    return 4 * M / N
```
:::


#### Solution

**calculating mean**


```python!
@delayed
def mean(*args):
    return sum(args) / len(args)

```

```python!
mean(1, 2, 3, 4).compute()
```


(NB): decorators can also be used as functions -- for instance, if you do not have access/do not want to change the source code of the function. Gives more flexibility.

```python!

def mean(*args):
    return sum(args) / len(args)

mean_lazy = delayed(mean)
```


**calculating pi**


```python!

import random

def calc_pi(N):
    """Computes the value of pi using N random samples."""
    M = 0
    for i in range(N):
        # take a sample
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y < 1.: M+=1
    return 4 * M / N
```


```python!
calc_pi_lazy = delayed(calc_pi)
```


```python!
n = 10**6
pi_p = mean(*[calc_pi_lazy(n) for _ in range(10)])

```

```python!
pi_p.visualize()
pi_p.compute()
```


Compare timing

```python!
pi_p = calc_pi_lazy(n)
```

```python!
%%timeit
pi_pi.compute()
```


```python!

pi_p = mean(*[calc_pi(n) for _ in range(10)])
```

```python!
%%timeit
pi_p.compute()
```


There is no speed-up because dask does not release the GIL. Can we use numba or numpy to release the GIL?
NB: this requires us to be able to modify the original function. If this is not possible, we need to use multiprocessing.

```python!

import random
import numba 

@numba.njit(nogil=True)
def calc_pi(N):
    """Computes the value of pi using N random samples."""
    M = 0
    for i in range(N):
        # take a sample
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y < 1.: M+=1
    return 4 * M / N


calc_pi_lazy = delayed(calc_pi)

```

Compare timing

```python!
pi_p = calc_pi_lazy(n)
```

```python!
%%timeit
pi_pi.compute()
```


```python!

pi_p = mean(*[calc_pi(n) for _ in range(10)])
```

```python!
%%timeit
pi_p.compute()
```


---


:::info
### Challenge: consider `pluck`
We previously discussed some generic operations on bags. In the documentation, lookup the `pluck` method. How would you implement this if `pluck` wasn’t there?

hint: Try `pluck` on some example data, for instance:
```python=
from dask import bags as db

data = [
   { "name": "John", "age": 42 },
   { "name": "Mary", "age": 35 },
   { "name": "Paul", "age": 78 },
   { "name": "Julia", "age": 10 }
]

bag = db.from_sequence(data)
...
```
:::

#### Solution

```python!
from dask import bags as db

data = [
   { "name": "John", "age": 42 },
   { "name": "Mary", "age": 35 },
   { "name": "Paul", "age": 78 },
   { "name": "Julia", "age": 10 }
]

bag = db.from_sequence(data)
```

`pluck` allows us to pick items from all tuples/dicts in a collection. 

`pluck` is a mapping.

```python!
def f(d):
    return d["name"]

bag.map(f).compute()
```

alternative: we use the built-in function `getitem` that implements the same functionality.

```python!
from operator import getitem 
```

```python!
getitem({"a": 1}, "a") # applies key to the dictionary and returns the value
```

```python!
bag.map(getitem, "name").compute()
```


compute mean of this collection? 

```python!
bag.map(getitem, "age").mean().compute()
```

or use a reduction with your own function. 
- be mindful that partitions may be of different sizes.
- we cannot pass argument to a reduction, so we need to work around this. for instance, we could use the `partial` approach below, or a lambda function, or wrap a given function into another function.


```python!
from functools import partial

def add(x, y):
    return x + y 

add_one = partial(f, 1)
```





---


## 🧠 Collaborative Notes

data parallelism -- execute the same instruction across input data

task parallelism -- execute different instructions across the input data


### Dask delayed 


![image](https://hackmd.io/_uploads/HJ9vSmORa.png)

Source: [Dask documentation](https://docs.dask.org/en/stable/10-minutes-to-dask.html)

By default, dask delayed uses threads to parallelize tasks.

```python!
from dask import delayed
```

delayed decorator: makes function "lazy", meaning it will not execute immediately.

```python!
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
    return result 


add(1, 2)
```


```python!
@delayed
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
    return result 

x_p = add(1, 2) # promise

```

How can we execute the promise? -- `compute`

```python!
x_p.compute()

```

We can combine promises to create arbitrary workflows, without running the workflow.

```python!
y_p = add(x_p, 3)

z_p = add(x_p, y_p)

```

Visualize the workflow
```python!
z_p.visualize(rankdir="LR") # show task graph from left to right

```

You may need graphviz to visualize. Use either of the two below, depending on how you set up your virtual environment/conda environment.
```python!
!pip install graphviz
!conda install python-graphviz

```
 
 Best practice: let the function create the data, and do not pass the data directly to the delayed function.
 If you need to load data and process them, you can write a delayed function to load the data and another delayed function to process the data.

In general, be as fine-grained as possible. Split tasks into separate functions, which allows to parallelize more easily.

You can also combine dask with numba. But using parallelization in numba and in dask is probably not a good idea because dask scheduling is not aware of how many threads numba is using, for instance.


You can also tell dask to use a specific scheduler.

```python!
x_p.compute(scheduler="threading") # this will use the python threading library

```

Do not combine dask arrays and dask delayed. Because dask arrays are already numpy arrays wrapped into delayed objects. In other words, dask array is a higher-level abstraction than dask delayed, and the two things should not be combined.


```python!
import dask.array as da 

```


```python!

arr = da.zeros((200, 100), chunks=(100, 100))

arr

```


```python!
arr_2 = arr + 1
arr_2.visualize()

```

```python!
arr_2

```

But you can convert an array to a delayed object (requires more understanding of how the delayed objects fit together)

```python!
arr_2.to_delayed()

```

Alternatively, you can also use `map_blocks()`

```python!
arr_2.map_blocs()

```


```python!

@delayed # creates a single delayed object with which one can do further computation
def gather(*args):
    return list(args)

def f(*args):
    print(args)

```


```python!
gathered = gather(*[add(i, i) for i in range(10)]) # the * unpacks the list

gathered.visualize()
```


```python!
gathered.compute() # tasks may not be executed in order

```


### Dask bag

dask bag is a different interface to dask. More tailored towards a certain type of workflows. 

What is a bag?
- computer science-speak for a collection of elements; the collection is not ordered
- this applies well for instance to workflows such as:
    - `map`: apply a function element-wise across all elements in the bag. `N` elements -> `N` elements.
    - `filter`: takes a conditional as input and returns elements that satisfy the condition. `N` elements -> <=`N` elements
    - `reduction`: aggregates elements in a bag. `N` elements -> `1` element. 
    - `flatten`: unpack a collection of collections into a single collection. `N` bags -> `1` bag
    - many more: https://docs.dask.org/en/stable/bag-api.html#bag-methods


By default, `bag` uses processes to parallelize tasks.

Remember that ordering is not guaranteed -- again, by definition, bags are not ordered.


```python!
import dask.bag as db 
```


```python!
bag = db.from_sequence(["mary", "had", "a", "little", "lamb"])
```


**map**

```python!
def upper(x):
    return x.upper()

upper(aba)
```


```python!
res = bag.map(upper)
```


```python!
res.compute()
```

dask bag uses partitions so that each partition contains about the same number of elements. the parallelization occurs across partitions. 


```python!
res.visualize()
```
Compared to `multiprocessing.map`, here we do not immediately compute the result. This allows us to build more complex workflows, and possibly better parallelization of multiple tasks.


**filter**


```python!
def pred(x):
    return "a" in x 


pred("mary")

```

```python!
res = bag.filter(pred)

```


```python!
res.visualize()
```

```python!
res.compute()
```


```python!
bag.map(pred).compute() # list of booleans
```


**reduction**

```python!
def count_chars(partition):
    return sum([len(word) for word in partition])

```


```python!
res = bag.reduction(count_chars, sum)

res.visualize()
```


```python!
res.compute()


```

**flatten**


```python!
bag.flatten.compute() 

```

for debugging, `take()` is useful
```python!
res = bag.flatten()
res.take(1)

```

how to take one partition? 

```python!
bag.map_partitions()
```


#### Last exercise done together


```python!
import random

def calc_pi(N):
    """Computes the value of pi using N random samples."""
    M = 0
    for i in range(N):
        # take a sample
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y < 1.: M+=1
    return 4 * M / N
```


```python!
n = 10**6

input = db.from_sequence([n]*10)
res = input.map(calc_pi).mean()
res.compute()

```

### AsyncIO


Designed for web-based applications - not for parallel computing, but it can be used for this application as well.

AsyncIO = "Async"hronous IO . But it is actually "synchronous".

#### Generators

Function: when "pure", given some input always gives the same output. Does not keep memory of ever being run.

Generators (and coroutine ) are different. See example:

```python=
def integers():
    a = 1
    while True:
        yield a
        a += 1

for i in integers():
    print(i)
    if i > 10:
        break
```

`integers` looks like a function but keep its state (=the point of the iteration where it got it)! "Encapsulate" the state, its hidden (and thus safer).

Generating iterators using Python with itertools:

```python=
from itertools import islice

list(islice(integers, 0, 10))

```
---

:::info
## Challenge: generate all even numbers
Can you write a generator that generates all even numbers? Try to reuse `integers()`. Extra: Can you generate the Fibonacci numbers?
:::

**Solution**

```python=
def even_integers():
    for i in integers():
        yield i * 2
        
# OR
evens = (i*2 for i in integers())
```

```python=
def fibonacci():
    a, b = 1, 1
    while True:
        yield a
        a = b
        b = a + b
```
---

"Yield" give control back to caller. Also can be used to "get" control:

```python=
def printer():
    # we start here
    while True:
        x = yield
        print(x)
```

```python=
p = printer()
next(p)
p.send("Mercury")
p.send("Venus")
p.send("Earth")
```

---

:::info
## Challenge: line numbers
Change `printer` to add line numbers to the output.
:::

**Solution**

```python=
def printer():
    lineno = 1
    while True:
        x = yield
        print(f"{lineno} {x}")
        lineno += 1
```

---

#### AsyncIO

```python=
import asyncio

async def counter(name):
    for i in range(5):
        print(f"{name:<10} {i:03}")
        await asyncio.sleep(0.2)
        
await counter("Venus")  # specific for Jupyter
# Venus      000
# Venus      001
# Venus      002
# Venus      003
# Venus      004
```

```python=
await counter("Venus")
await counter("Earth")
```

Collect output together:

```python=
await asyncio.gather(
    counter("Venus"),
    counter("Earth"),
)
```

If you need to run coroutine outside Jupyter:
```python=
import asyncio

async def main():
    ...
    
if __name__ == "__main__":
    asyncio.run(main())
```

The following cells are used to time async code:

```python=
from dataclasses import dataclass
from typing import Optional
from time import perf_counter
from contextlib import asynccontextmanager


@dataclass
class Elapsed:
    time: Optional[float] = None


@asynccontextmanager
async def timer():
    e = Elapsed()
    t = perf_counter()
    yield e
    e.time = perf_counter() - t
```

Use it as:
```python=
async with timer() as t:
    await asyncio.sleep(0.2)
print(f"Took {t.time} seconds.")    
```

#### Compute $\pi$

Again, we will estimate the value of pi, now using `asyncio`:

```python=
import random
import numba


@numba.njit(nogil=True)
def calc_pi(N):
    M = 0
    for i in range(N):
        # Simulate impact coordinates
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        # True if impact happens inside the circle
        if x**2 + y**2 < 1.0:
            M += 1
    return 4 * M / N
```

To start a coroutine on a new thread:
```python=
async with timer() as t:
    pi = await asyncio.to_thread(calc_pi, 10**7)
    print(f"pi = {pi}")
t.time
```

---

:::info
## Gather multiple outcomes
We've seen that we can gather multiple coroutines using `asyncio.gather`. Now gather several `calc_pi` computations, and time them.
:::

**Solution**

```python=
async with timer() as t:
    pi = await asyncio.gather(   
    *(asyncio.to_thread(calc_pi, 10**7) for _ in range(10))
    )
    print(f"pi = {pi}")
t.time
```


---

### Mandelbrot fractal

$$z_{i} = z^2_{i-1} + c$$
$$z_0 = 0$$

For some values of $c$, the iteration diverges. 
Testing the following for different values of c will show whether the iteration diverges / converges / is bound.

```python=
def mandelbrot_orbit(c):
    z = 0 + 0.j
    while True:
        yield z
        z = z**2 + c
```

Let's calculate it for values in the complex plane:

```python=
max_iter = 256
width = 256
height = 256
center = -0.8 + 0.j
extent = 3. + 3.j
scale = max((extent / width).real, (extent / height).imag)
```

For the parameters above, calculate the number of iterations for which it diverges:

```python=
import numpy as np

result = np.zeros((height, width), int)
for j in range(height):
    for i in range(width):
        c = center + (i - width // 2 + (j - height // 2)*1j) * scale
        z = 0
        for k in range(max_iter):
            z = z**2 + c
            if (z * z.conjugate()).real > 4.0:
                break
        result[j, i] = k
```

```
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plot_extent = (width + 1j * height) * scale
z1 = center - plot_extent / 2
z2 = z1 + plot_extent
ax.imshow(result**(1/3), origin='lower', extent=(z1.real, z2.real, z1.imag, z2.imag))
ax.set_xlabel("$\Re(c)$")
ax.set_ylabel("$\Im(c)$")
```

How can we make the code computing the fractal faster?

To make the problem "heavier", you can increase the values of `width`, `height`. You can also change the center and number of iterations (to get a better contrast):

```python=
max_iter = 1024
center = -1.1195+0.2718j
extent = 0.005+0.005j
```

## Feedback

### Morning

🎉 What went well?
- Interesting material
- Nice examples by redoing the same task with different dask functions
- good method overview
- Good to start from low level & build up to higher level methods


💡 What could be better?

- 'real examples'? I know Pi by now 
- +1 for comment above
- struggled a lot with the visualisation method, installing, reinstalling and so on did not work. I know it's hard to anticipate but may be worthwhile to include in the installation tests and leave it as optional in material and exercises? 


### Afternoon

🎉 What went well?

- Good explanation of the asyncio magic, fun problem
- Interesting introduction to asyncio
- Dask delayed and bag 
- quite liked all tricks, thanks


💡 What could be better?

- Maybe choose one of the possible parallelisation methods and remove it. Then use the time to offer more complicated examples or offer guidance on which method to use when. Right now we might have too many options.
- Maybe a better outlook/summary of how we can apply the different methods/what their strenghts/weaknesses are?
- A recap of all methods used, timed to compare performances could've been a nice wrap-up. Still was pretty good. Thanks!

## Final Survey 

Post workshop survey: https://www.surveymonkey.com/r/YYFKVBD

## 📚 Resources
- [dask bag methods](https://docs.dask.org/en/stable/bag-api.html#bag-methods)


### Catch-up from day 1: collecting results from threads

- for instance, [this method](https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread) is not easy to compile to numba
    - we could use a class (as in the above link), but I have not tried this
- problem: it's hard to make numba modify objects like lists, which is necessary in threaded 
    - in the above approach, each thread would write the result into a list at different indices. but we cannot numba-compile lists
- we can use `concurrent.futures` instead, which provides an easier-to-use interface
- under the hood it does not do quite the same as `threading.Thread`: it asynchronously executes the function. But Johan will certainly be able to tell you more.

Documentation: https://docs.python.org/3/library/concurrent.futures.html#executor-objects

https://stackoverflow.com/questions/61351844/difference-between-multiprocessing-asyncio-threading-and-concurrency-futures-i

key idea: "executor" are what was "Pool" in multiprocessing yesterday, and it works for threads and processes. 

```python!
import numba
import random 

@numba.jit(nopython=True, nogil=True)
def calc_pi(N, name=None):
    printing = name is not None 
    if printing:
        print(f"{name}: starting")
    M = 0 
    for i in range(N):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x**2 + y**2 < 1:
            M += 1
    
    if printing:
        print(f"{name}: Done")
            
    return 4*M/N
    

calc_pi(100) # compile

```


```python!
from  concurrent.futures  import ThreadPoolExecutor
```


```python!
#show the result
executor = ThreadPoolExecutor(max_workers=5)
a = executor.map(calc_pi, [10**7//2, 10**7//2])
list(a) # a is a generator
```


```python!
%%timeit -n 10 -r 10
n_chunks = 2
executor = ThreadPoolExecutor(max_workers=5)
work = [10**7//n_chunks for _ in range(n_chunks)]
a = executor.map(calc_pi, work)

```


```python!
%%timeit -n 10 -r 10
a = calc_pi(10**7)


```

**Doing the same with processes**

```python!

from concurrent.futures import ProcessPoolExecutor
```


```python!

# show the result 
executor = ProcessPoolExecutor(max_workers=4)
n_chunks = 2
work = [10**7//n_chunks for _ in range(n_chunks)]
a = executor.map(calc_pi, work)
```

```python!

%%timeit -n 10 -r 10
executor = ProcessPoolExecutor(max_workers=4)

n_chunks = 2
work = [10**7//n_chunks for _ in range(n_chunks)]
a = executor.map(calc_pi, work)

```

