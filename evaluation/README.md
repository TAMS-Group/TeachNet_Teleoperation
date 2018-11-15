### ShadowTeleop eval
0. generate predict file `input.csv` by run main.py(uncomment some code in `test()`)
1. put generated joint predict file `input.csv` in `predict`.

2. convert joint to xyz(`output.csv`)
```bash
$ roslaunch shadow_vision_telelop shadow_fk.launch
```

3. convert xyz into uvd(remember to change the path in the script)
```bash
$ python cartesian2uvd.py
```
4. plot result
```bash
$ python eval.py <predict_pkl> <label_pkl>
```
