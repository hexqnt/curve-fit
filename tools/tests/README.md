# UI-проверки

Все сценарии запускают полное приложение через общий `egui_kittest` harness из `tools/test-support`. Приложение создаётся с английским языком, фиксированными ОС, размером окна и шагом времени; действия выполняются через UI, состояние доступно только для чтения.

Запуск из корня проекта:

```sh
cargo test -p curve-fit-ui-tests --test ui
```

Проверки Fit/Stop используют `harness_with_paused_fit`: первый фоновый расчёт ждёт отпускания `FitWorkerPause`. Это позволяет проверить состояние расчёта и блокировку полей независимо от скорости машины. Удаление объекта паузы, в том числе при панике теста, отпускает поток. Последующие расчёты работают обычным образом. В сборке без feature `testing` этой инфраструктуры нет.

На Linux, Windows и macOS выполняются одинаковые проверки взаимодействий и состояния приложения. Эталонные скриншоты и графический renderer для тестов не используются.

Сборки приложения без тестовой инфраструктуры проверяются отдельно, чтобы объединение features в workspace не скрывало зависимость от `testing`:

```sh
cargo build -p curve-fit --no-default-features
cargo build -p curve-fit --no-default-features --target wasm32-unknown-unknown
cargo clippy -p curve-fit --no-default-features --target wasm32-unknown-unknown -- -D warnings
```
