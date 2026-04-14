/// Аналитическая функция потерь по предсказанию модели.
pub(crate) trait PredictionLoss {
    fn value(&self, prediction: f64, target: f64) -> f64;
    fn d_prediction(&self, prediction: f64, target: f64) -> f64;
    fn d2_prediction(&self, prediction: f64, target: f64) -> f64;
}
