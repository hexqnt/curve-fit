# curve-fit

`curve-fit` — учебное приложение для подбора параметров кривой по набору точек. Основаня цель приложения это наработка интуиции по модельным кривым.

[![CI](https://github.com/hexqnt/curve-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/hexqnt/curve-fit/actions/workflows/ci.yml)

![Alt text](images/curve-fit-screenshot.png "Optional title text")

Отличие десктопной версии от web-версии:

- чуть выше производительность
- нет подвисания при обучении(фитинге кривой)

## Run Desktop

```bash
cargo run
```

## Run Web (wasm)

1. Установить таргет:

```bash
rustup target add wasm32-unknown-unknown
```

1. Установить `trunk` (если не установлен):

```bash
cargo install trunk
```

1. Запустить web-версию:

```bash
trunk serve
```
