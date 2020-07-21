# How can we download JSUT dataset?

[https://sites.google.com/site/shinnosuketakamichi/publication/jsut](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

After donwloading, please extract at this directory and specify `jsut_path` in `preprocess.sh`

### What units does the duration of *.lab file use?

In *.lab files, we see the label as follows:

`0 3125000 xx^xx-sil+m=i/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:02+xx_xx/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx/G:3_3%0_xx_xx/H:xx_xx/I:xx-xx@xx+xx&xx-xx|xx+xx/J:5_23/K:1+5-23`

ANSWER:

Basically, the unit is second.
To convert second, we should use below calculation.

`raw / 10000000.0`