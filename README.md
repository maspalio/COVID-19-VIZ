# COVID-19

Jupyter notebook as per 2019 Novel Coronavirus COVID-19 (2019-nCoV) Data Repository by Johns Hopkins CSSE (https://github.com/CSSEGISandData/COVID-19)

## Docker

```bash
docker build --tag covid-19-viz .
docker run --interactive --tty --publish 8888:8888 covid-19-viz:latest
```
