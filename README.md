# Cr(VI) removal predictor

Static browser application for the final sulfur-autotrophic Cr(VI) removal GPR model (`CrVI-GPR-2026-08-31`). Open `index.html` directly; inputs and predictions remain in the browser.

## Model scope

- Inputs: reaction time, inoculum, initial Cr(VI), temperature, sulfate, and nitrate.
- Training data: 172 positive-time condition means from 18 environmental conditions.
- Validation: nested grouped cross-validation (`OOF R2 = 0.789`, overall `MAE = 0.077`, condition-equal `MAE = 0.0766`).
- The displayed trajectory uses a cumulative-maximum post-process. This does not make the underlying GPR monotonic.
- The mean +/- 1.96 standard deviation range is not calibrated; grouped OOF coverage was 76.7%.
- Most multifactor combinations were not directly observed because the source experiment used an OFAT design.

## Files

- `model-data.js`: exported scaler, Matern GPR parameters, training representation, and reference cases.
- `app.js`: local prediction, uncertainty, applicability checks, kinetics, and result export.
- `styles.css`: responsive interface styles.
- `tests/ui_smoke.js`: numerical and responsive browser smoke test.
