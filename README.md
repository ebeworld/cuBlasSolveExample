#Solving Lnear Equations Using CUDA

## Enlist code

```bash
https://github.com/ebeworld/cuBlasSolveExample.git
```

## Build

rmdir /s /q build
mkdir build
cd build

cmake ..
cmake --build . --config Release

## Execute

build\Release\solve_linear_equation.exe

## Result

In this code I chose 

```bash
|3  1 | x | x1 | = | 9 |
|1  2 |   | x2 |   | 8 |
```

which is 

``` bash
3x1 +  x2 = 9
 x1 + 2x2 = 8
```

Expected solution:

``` bash
Solution:
x1 = 2
x2 = 3
```
