# AlbOpt - Album Optimization

Selects and weights the most unique and informative photos out of a larger album.

![example collage](./sample.png "Example Collage")

# Example Usage:

AlbOpt requires python 3.8, after that install all requirements from requirements.txt

```bash
$ python albopt --files-pattern 'my_pics/*' -r 12 \
  --output-name report/albopt --min-ratio 0.2 \
  --algo albopt_pow2
```

for more options see `python albopt --help`
