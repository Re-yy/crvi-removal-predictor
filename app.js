(function () {
  "use strict";

  const model = window.CRVI_MODEL;
  if (!model) {
    document.body.innerHTML = "<p style='padding:2rem'>模型数据未加载，请确认 model-data.js 与本页面位于同一目录。</p>";
    return;
  }

  const uiFeatures = ["time", "inoculum", "initial-cr", "temperature", "sulfate", "nitrate"];
  const modelKeys = ["time", "inoculum", "initial_cr", "temperature", "sulfate", "nitrate"];
  const environmentKeys = ["inoculum", "initial_cr", "temperature", "sulfate", "nitrate"];
  const featureLabels = {
    time: "反应时间",
    inoculum: "接种量",
    initial_cr: "初始 Cr(VI)",
    temperature: "温度",
    sulfate: "初始硫酸盐",
    nitrate: "初始硝酸盐",
  };
  const defaults = { time: 48, inoculum: 10, "initial-cr": 10, temperature: 30, sulfate: 0, nitrate: 0 };
  const uiDomain = {
    time: { min: 0, max: 240 },
    inoculum: { min: 0.01, max: 0.20 },
    initial_cr: { min: 0.1, max: 100 },
    temperature: { min: 0, max: 50 },
    sulfate: { min: 0, max: 1000 },
    nitrate: { min: 0, max: 1000 },
  };
  const circumference = 2 * Math.PI * 86;
  let updateFrame = null;
  let lastState = null;

  const byId = (id) => document.getElementById(id);
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
  const nearlyEqual = (left, right, tolerance = 1e-9) => Math.abs(left - right) <= tolerance;

  function modelVector(input) {
    return modelKeys.map((key) => Number(input[key]));
  }

  function scaledVector(input) {
    return modelVector(input).map((value, index) =>
      (value - model.scaler.mean[index]) / model.scaler.scale[index]
    );
  }

  function matern32(left, right) {
    let squaredRadius = 0;
    for (let index = 0; index < left.length; index += 1) {
      const delta = (left[index] - right[index]) / model.kernel.lengthScale[index];
      squaredRadius += delta * delta;
    }
    const radius = Math.sqrt(squaredRadius);
    const scaledRadius = Math.sqrt(3) * radius;
    return model.kernel.constantValue * (1 + scaledRadius) * Math.exp(-scaledRadius);
  }

  function forwardSolve(lower, rightHandSide) {
    const solution = new Array(rightHandSide.length).fill(0);
    for (let row = 0; row < rightHandSide.length; row += 1) {
      let residual = rightHandSide[row];
      for (let column = 0; column < row; column += 1) {
        residual -= lower[row][column] * solution[column];
      }
      solution[row] = residual / lower[row][row];
    }
    return solution;
  }

  function predictRaw(input, includeUncertainty = true) {
    const x = scaledVector(input);
    const covariance = model.gpr.xTrainScaled.map((row) => matern32(x, row));
    let standardizedMean = 0;
    for (let index = 0; index < covariance.length; index += 1) {
      standardizedMean += covariance[index] * model.gpr.alpha[index];
    }

    const rawMean = standardizedMean * model.gpr.yStd + model.gpr.yMean;
    let mean = clamp(rawMean, 0, 1);
    let std = null;
    let lower95 = null;
    let upper95 = null;

    if (includeUncertainty) {
      const solved = forwardSolve(model.gpr.lowerCholesky, covariance);
      const explainedVariance = solved.reduce((sum, value) => sum + value * value, 0);
      const scaledVariance = Math.max(
        0,
        model.kernel.constantValue + model.kernel.noiseLevel - explainedVariance
      );
      std = Math.sqrt(scaledVariance) * model.gpr.yStd;
      lower95 = clamp(rawMean - 1.96 * std, 0, 1);
      upper95 = clamp(rawMean + 1.96 * std, 0, 1);
    }

    if (input.time <= 0) {
      mean = 0;
      lower95 = includeUncertainty ? 0 : null;
      upper95 = includeUncertainty ? 0 : null;
    }
    return { mean, rawMean, std, lower95, upper95 };
  }

  function predict(input, includeUncertainty = true) {
    const requestedTime = Number(input.time);
    if (requestedTime <= 0) {
      return { ...predictRaw({ ...input, time: 0 }, includeUncertainty), requestedTime: 0, sourceTime: 0 };
    }

    let bestTime = 0;
    let bestMean = 0;
    const wholeHours = Math.floor(requestedTime);
    for (let time = 1; time <= wholeHours; time += 1) {
      const candidate = predictRaw({ ...input, time }, false).mean;
      if (candidate > bestMean) {
        bestMean = candidate;
        bestTime = time;
      }
    }
    if (requestedTime > wholeHours) {
      const candidate = predictRaw({ ...input, time: requestedTime }, false).mean;
      if (candidate > bestMean) {
        bestMean = candidate;
        bestTime = requestedTime;
      }
    }

    const result = predictRaw({ ...input, time: bestTime }, includeUncertainty);
    return { ...result, mean: bestMean, requestedTime, sourceTime: bestTime };
  }

  function collectInput() {
    return {
      time: Number(byId("time-number").value),
      inoculum: Number(byId("inoculum-number").value) / 100,
      initial_cr: Number(byId("initial-cr-number").value),
      temperature: Number(byId("temperature-number").value),
      sulfate: Number(byId("sulfate-number").value),
      nitrate: Number(byId("nitrate-number").value),
    };
  }

  function validate(input) {
    for (const key of modelKeys) {
      const value = input[key];
      const domain = uiDomain[key];
      if (!Number.isFinite(value)) return `${featureLabels[key]}必须是有效数值。`;
      if (value < domain.min || value > domain.max) {
        const scale = key === "inoculum" ? 100 : 1;
        const unit = key === "inoculum" ? "% (v/v)" : "";
        return `${featureLabels[key]}应位于 ${domain.min * scale}–${domain.max * scale}${unit}。`;
      }
    }
    return "";
  }

  function buildObservedConditions() {
    const candidates = [{ ...model.baseline }];
    for (const key of environmentKeys) {
      for (const level of model.domain[key].levels) {
        candidates.push({ ...model.baseline, [key]: level });
      }
    }
    const unique = new Map();
    for (const condition of candidates) {
      const signature = environmentKeys.map((key) => condition[key]).join("|");
      unique.set(signature, condition);
    }
    return [...unique.values()];
  }

  const observedConditions = buildObservedConditions();

  function scaledConditionDistance(left, right) {
    const squared = environmentKeys.reduce((sum, key) => {
      const width = Math.max(model.domain[key].max - model.domain[key].min, 1e-12);
      return sum + ((left[key] - right[key]) / width) ** 2;
    }, 0);
    return Math.sqrt(squared);
  }

  function assessApplicability(input) {
    const outside = modelKeys.filter((key) =>
      input[key] < model.domain[key].min || input[key] > model.domain[key].max
    );
    const nearest = observedConditions
      .map((condition) => ({ condition, distance: scaledConditionDistance(input, condition) }))
      .sort((left, right) => left.distance - right.distance)[0];
    const exactEnvironment = nearest.distance < 1e-9;
    const exactTime = model.domain.time.levels.some((level) => nearlyEqual(level, input.time));
    const changedFactors = environmentKeys.filter((key) => !nearlyEqual(input[key], model.baseline[key]));

    if (outside.length) {
      return {
        level: "outside",
        title: "超出训练数值范围",
        detail: `${outside.map((key) => featureLabels[key]).join("、")}位于训练域之外；结果是数值外推，风险高于折外性能所描述的范围。`,
        nearest,
        outside,
        changedFactors,
      };
    }
    if (exactEnvironment && exactTime) {
      return {
        level: "observed",
        title: "实验输入已覆盖",
        detail: "环境组合和采样时间均出现在实验记录中；页面数值仍是模型预测，不替代原始测量。",
        nearest,
        outside,
        changedFactors,
      };
    }
    if (exactEnvironment) {
      return {
        level: "interpolation",
        title: "已测环境内的时间插值",
        detail: "环境条件已有实验轨迹支持，但当前时间点未直接采样。",
        nearest,
        outside,
        changedFactors,
      };
    }
    if (changedFactors.length > 1) {
      return {
        level: "multifactor",
        title: "未实测多因素组合",
        detail: `${changedFactors.length}个环境因素同时偏离基准。源数据为OFAT设计，该结果不能证明协同、拮抗或最优工况。`,
        nearest,
        outside,
        changedFactors,
      };
    }
    return {
      level: "interpolation",
      title: "训练范围内的条件插值",
      detail: "各变量位于训练数值范围内，但这一精确环境组合未被直接测量。",
      nearest,
      outside,
      changedFactors,
    };
  }

  function buildTrajectory(input) {
    const trajectory = [];
    let cumulativeMaximum = 0;
    for (let time = 0; time <= uiDomain.time.max; time += 1) {
      const raw = predictRaw({ ...input, time }, false).mean;
      cumulativeMaximum = Math.max(cumulativeMaximum, raw);
      trajectory.push({ time, raw, mean: cumulativeMaximum });
    }
    return trajectory;
  }

  function thresholdTime(trajectory, threshold) {
    const point = trajectory.find((item) => item.mean >= threshold);
    return point ? point.time : null;
  }

  function updateGauge(result, rawRequested) {
    const percent = result.mean * 100;
    byId("prediction-percent").textContent = `${percent.toFixed(1)}%`;
    byId("prediction-fraction").textContent = result.mean.toFixed(3);
    byId("prediction-interval").textContent = `${(result.lower95 * 100).toFixed(1)}% – ${(result.upper95 * 100).toFixed(1)}%`;
    byId("raw-prediction").textContent = `${(rawRequested.mean * 100).toFixed(1)}%`;
    byId("display-rule").textContent = result.sourceTime < result.requestedTime
      ? `累计最大值（取自 ${result.sourceTime} h）`
      : "累计最大值后处理";
    const progress = byId("gauge-progress");
    progress.style.strokeDasharray = String(circumference);
    progress.style.strokeDashoffset = String(circumference * (1 - result.mean));
    byId("gauge-desc").textContent = `页面显示预测去除率${percent.toFixed(1)}%，GPR未校准预测范围${(result.lower95 * 100).toFixed(1)}%到${(result.upper95 * 100).toFixed(1)}%。`;
  }

  function updateSupportFields(input) {
    for (const key of modelKeys) {
      const field = document.querySelector(`.field[data-feature="${key}"]`);
      const outside = input[key] < model.domain[key].min || input[key] > model.domain[key].max;
      field.classList.toggle("outside-support", outside);
    }
  }

  function svgElement(name, attributes = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", name);
    Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
  }

  function drawCurve(input, trajectory, selectedResult) {
    const svg = byId("curve-chart");
    svg.replaceChildren();
    const width = 700;
    const height = 280;
    const margin = { left: 55, right: 22, top: 18, bottom: 42 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const curveMax = uiDomain.time.max;
    const x = (time) => margin.left + (time / curveMax) * plotWidth;
    const y = (value) => margin.top + (1 - clamp(value, 0, 1)) * plotHeight;

    const trainedTimeMax = model.domain.time.max;
    svg.appendChild(svgElement("rect", {
      x: x(trainedTimeMax), y: margin.top,
      width: x(curveMax) - x(trainedTimeMax), height: plotHeight,
      fill: "rgba(181,85,42,0.055)",
    }));
    svg.appendChild(svgElement("line", {
      x1: x(trainedTimeMax), y1: margin.top, x2: x(trainedTimeMax), y2: y(0),
      stroke: "#b5552a", "stroke-width": 1, "stroke-dasharray": "3 5",
    }));
    const extensionLabel = svgElement("text", {
      x: x(trainedTimeMax) + 7, y: margin.top + 13,
      fill: "#8a604c", "font-size": 10,
    });
    extensionLabel.textContent = "时间外推";
    svg.appendChild(extensionLabel);

    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      const yPosition = y(tick);
      svg.appendChild(svgElement("line", {
        x1: margin.left, y1: yPosition, x2: width - margin.right, y2: yPosition,
        stroke: "#d7ddd8", "stroke-width": 1,
      }));
      const label = svgElement("text", {
        x: margin.left - 10, y: yPosition + 4, "text-anchor": "end",
        fill: "#66726d", "font-size": 12,
      });
      label.textContent = `${Math.round(tick * 100)}`;
      svg.appendChild(label);
    }
    for (const tick of [0, 40, 80, 120, 160, 200, 240]) {
      const label = svgElement("text", {
        x: x(tick), y: height - 14, "text-anchor": "middle",
        fill: "#66726d", "font-size": 12,
      });
      label.textContent = tick;
      svg.appendChild(label);
    }

    const sampled = trajectory.filter((point) => point.time % 4 === 0 || point.time === curveMax);
    const points = sampled.map((point) => [x(point.time), y(point.mean)]);
    const areaPath = `M ${points[0][0]} ${y(0)} `
      + points.map(([px, py]) => `L ${px} ${py}`).join(" ")
      + ` L ${points.at(-1)[0]} ${y(0)} Z`;
    svg.appendChild(svgElement("path", { d: areaPath, fill: "rgba(18,111,99,0.10)" }));
    svg.appendChild(svgElement("polyline", {
      points: points.map(([px, py]) => `${px},${py}`).join(" "),
      fill: "none", stroke: "#126f63", "stroke-width": 3,
      "stroke-linecap": "round", "stroke-linejoin": "round",
    }));
    svg.appendChild(svgElement("line", {
      x1: x(input.time), y1: margin.top, x2: x(input.time), y2: y(0),
      stroke: "#c58b00", "stroke-width": 1.5, "stroke-dasharray": "5 5",
    }));
    svg.appendChild(svgElement("circle", {
      cx: x(input.time), cy: y(selectedResult.mean), r: 6,
      fill: "#f2b705", stroke: "#24342f", "stroke-width": 2,
    }));
    const xLabel = svgElement("text", {
      x: width / 2, y: height - 1, "text-anchor": "middle",
      fill: "#34453f", "font-size": 12,
    });
    xLabel.textContent = "反应时间 (h)";
    svg.appendChild(xLabel);
    const yLabel = svgElement("text", {
      x: 15, y: height / 2, transform: `rotate(-90 15 ${height / 2})`,
      "text-anchor": "middle", fill: "#34453f", "font-size": 12,
    });
    yLabel.textContent = "预测去除率 (%)";
    svg.appendChild(yLabel);
    byId("curve-condition").textContent = `接种 ${(input.inoculum * 100).toFixed(1)}% · Cr(VI) ${input.initial_cr} mg/L · ${input.temperature} °C`;
  }

  function updateThresholds(trajectory) {
    const t50 = thresholdTime(trajectory, 0.5);
    const t90 = thresholdTime(trajectory, 0.9);
    byId("t50-value").textContent = t50 === null ? "未达到" : `${t50} h`;
    byId("t90-value").textContent = t90 === null ? "未达到" : `${t90} h`;
  }

  function setUiValues(values) {
    for (const [feature, value] of Object.entries(values)) {
      byId(`${feature}-range`).value = value;
      byId(`${feature}-number`).value = value;
    }
  }

  function render() {
    const input = collectInput();
    const error = validate(input);
    byId("form-error").hidden = !error;
    byId("form-error").textContent = error;
    if (error) return false;

    const result = predict(input);
    const rawRequested = predictRaw(input);
    const applicability = assessApplicability(input);
    const trajectory = buildTrajectory(input);
    updateGauge(result, rawRequested);
    updateSupportFields(input);
    drawCurve(input, trajectory, result);
    updateThresholds(trajectory);
    lastState = { input, result, rawRequested, applicability, trajectory };
    return true;
  }

  function scheduleRender() {
    if (updateFrame) cancelAnimationFrame(updateFrame);
    updateFrame = requestAnimationFrame(() => {
      updateFrame = null;
      render();
    });
  }

  function synchronize(sourceId, targetId) {
    const source = byId(sourceId);
    const target = byId(targetId);
    source.addEventListener("input", () => {
      target.value = source.value;
      scheduleRender();
    });
  }

  function downloadResult() {
    if (!lastState && !render()) return;
    const { input, result, rawRequested, applicability } = lastState;
    const payload = {
      model: {
        name: model.modelName,
        version: model.modelVersion,
        validation: model.performance,
        trainingRows: model.trainingRows,
        environmentalConditions: model.environmentalConditions,
        sourceSha256: model.sourceSha256,
      },
      input,
      prediction: {
        displayedRemovalFraction: result.mean,
        displayedRemovalPercent: result.mean * 100,
        rawGprAtRequestedTimeFraction: rawRequested.mean,
        rawGprAtRequestedTimePercent: rawRequested.mean * 100,
        monotonicSourceTimeHours: result.sourceTime,
        modelStandardDeviationFraction: result.std,
        uncalibratedModelRangeFraction: [result.lower95, result.upper95],
        intervalOofCoverage: model.performance.intervalOOFCoverage,
      },
      applicability: {
        status: applicability.title,
        level: applicability.level,
        explanation: applicability.detail,
        outsideTrainingRange: applicability.outside,
        nonBaselineFactors: applicability.changedFactors,
        nearestObservedCondition: applicability.nearest.condition,
        nearestScaledDistance: applicability.nearest.distance,
      },
      interpretation: "科研展示与候选实验筛选结果；未校准区间不保证95%覆盖，未实测多因素组合不构成因果或交互证据。",
      generatedAt: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `crvi-prediction-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  for (const feature of uiFeatures) {
    synchronize(`${feature}-range`, `${feature}-number`);
    synchronize(`${feature}-number`, `${feature}-range`);
  }

  byId("prediction-form").addEventListener("submit", (event) => {
    event.preventDefault();
    render();
  });

  byId("reset-button").addEventListener("click", () => {
    setUiValues(defaults);
    render();
  });

  byId("download-button").addEventListener("click", downloadResult);
  byId("model-version").textContent = model.modelVersion;
  window.__CRVI_TEST__ = {
    predict,
    predictRaw,
    collectInput,
    validate,
    assessApplicability,
    buildTrajectory,
    observedConditions,
  };
  render();
}());
