# curve-fit

[🇺🇸 English](./README.md) · [🇷🇺 Русский](./README.ru.md)

[![CI](https://github.com/hexqnt/curve-fit/actions/workflows/ci.yml/badge.svg)](https://github.com/hexqnt/curve-fit/actions/workflows/ci.yml)

`curve-fit` is an educational application for fitting curves to real or synthetic data. It provides an interactive way to explore how different models, optimizers, and loss functions affect the result and convergence process.

Try the [web version](https://curve-fit.hexq.ru) or install the desktop application for better performance on larger datasets.

![curve-fit screenshot](images/curve-fit-screenshot.png)

## Features

- Parametric models, including polynomial, exponential, sigmoid, peak, power, bi-exponential, and damped oscillation models.
- Linear, PCHIP, natural cubic, and Akima splines.
- Multiple optimizers and loss functions.
- Interactive plots, iteration diagnostics, and data import and export.

`curve-fit` is designed as a learning and experimentation tool, not as a general-purpose production fitting library.

## Install a prebuilt release

Download the archive for your platform from the [latest GitHub release](https://github.com/hexqnt/curve-fit/releases/latest):

| Platform            | Archive suffix       | Executable      |
| ------------------- | -------------------- | --------------- |
| Linux x86-64        | `linux-x86_64.zip`   | `curve-fit`     |
| macOS Apple silicon | `macos-aarch64.zip`  | `curve-fit`     |
| Windows x86-64      | `windows-x86_64.zip` | `curve-fit.exe` |

Extract the archive, then launch the executable. On Linux or macOS, you can run it from a terminal:

```bash
chmod +x curve-fit
./curve-fit
```

On Windows, double-click `curve-fit.exe` or run it from PowerShell:

```powershell
.\curve-fit.exe
```

If macOS blocks the first launch, allow the application in **System Settings → Privacy & Security**, then open it again.

## Install from source

Install [Rust](https://rustup.rs), then install and run `curve-fit` with the nightly toolchain:

```bash
rustup toolchain install nightly
cargo +nightly install --git https://github.com/hexqnt/curve-fit --locked
curve-fit
```

On Linux, compiling may also require the X11, Wayland, and OpenGL development packages provided by your distribution.

## Run the web version locally

Clone the repository, then run:

```bash
rustup target add --toolchain nightly wasm32-unknown-unknown
cargo install trunk --locked
trunk serve --open
```
