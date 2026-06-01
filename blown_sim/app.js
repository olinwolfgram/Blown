const STORAGE_KEY = "blownWingCesiumIonToken";
const DEG2RAD = Math.PI / 180.0;
const EARTH_RADIUS_M = 6378137.0;
const MODEL_HEADING_CORRECTION_DEG = 0.0;
const MODEL_PITCH_CORRECTION_DEG = 90.0;
const MODEL_ROLL_CORRECTION_DEG = 180.0;
const PROP_VISUAL_RPM_SCALE = 0.005;
const PROP_ROTATION_AXIS = "z";

const PROP_METADATA = [
    { name: "PROP_OCC_01", side: "right", spinSign: +1 },
    { name: "PROP_OCC_02", side: "left", spinSign: -1 },
    { name: "PROP_OCC_03", side: "left", spinSign: +1 },
    { name: "PROP_OCC_04", side: "left", spinSign: -1 },
    { name: "PROP_OCC_05", side: "right", spinSign: +1 },
    { name: "PROP_OCC_06", side: "right", spinSign: -1 },
    { name: "PROP_OCC_07", side: "right", spinSign: +1 },
    { name: "PROP_OCC_08", side: "left", spinSign: -1 },
    { name: "PROP_OCC_09", side: "right", spinSign: +1 },
    { name: "PROP_OCC_10", side: "left", spinSign: -1 },
];

const defaultOrigin = {
    latDeg: 37.423273,
    lonDeg: -122.176076,
    altM: 40.0,
};

const ui = {
    ionToken: document.getElementById("ionToken"),
    saveTokenBtn: document.getElementById("saveTokenBtn"),
    originLat: document.getElementById("originLat"),
    originLon: document.getElementById("originLon"),
    originAlt: document.getElementById("originAlt"),
    csvFile: document.getElementById("csvFile"),
    visualizeBtn: document.getElementById("visualizeBtn"),
    status: document.getElementById("status"),
    rowCount: document.getElementById("rowCount"),
    durationVal: document.getElementById("durationVal"),
    controllerVal: document.getElementById("controllerVal"),
    telemetryTime: document.getElementById("telemetryTime"),
    telemetryLat: document.getElementById("telemetryLat"),
    telemetryLon: document.getElementById("telemetryLon"),
    telemetryAlt: document.getElementById("telemetryAlt"),
    telemetryRoll: document.getElementById("telemetryRoll"),
    telemetryPitch: document.getElementById("telemetryPitch"),
    telemetryYaw: document.getElementById("telemetryYaw"),
    telemetryUVW: document.getElementById("telemetryUVW"),
    telemetryPQR: document.getElementById("telemetryPQR"),
    telemetryRpmLR: document.getElementById("telemetryRpmLR"),
    telemetrySurfaces: document.getElementById("telemetrySurfaces"),
    hudText: document.getElementById("hudText"),
    sidebarMotorBars: document.getElementById("sidebarMotorBars"),
    sidebarSurfaceBars: document.getElementById("sidebarSurfaceBars"),
    sidebarMotorScaleText: document.getElementById("sidebarMotorScaleText"),
    sidebarSurfaceScaleText: document.getElementById("sidebarSurfaceScaleText"),
};

let viewer = null;
let activeEntity = null;
let activeRows = [];
let activeOrigin = { ...defaultOrigin };
let tickHandler = null;
let activeControlHud = null;

window.addEventListener("error", (event) => {
    if (ui.status) {
        ui.status.textContent = `App error: ${event.message}`;
        ui.status.style.color = "#ff7b72";
    }
});

window.addEventListener("unhandledrejection", (event) => {
    if (ui.status) {
        ui.status.textContent = `Promise error: ${event.reason?.message || event.reason || "unknown error"}`;
        ui.status.style.color = "#ff7b72";
    }
});

function formatNumber(value, digits = 2) {
    if (!Number.isFinite(value)) {
        return "--";
    }
    return value.toFixed(digits);
}

function setStatus(message, kind = "neutral") {
    ui.status.textContent = message;
    const colors = {
        neutral: "#d0d7de",
        success: "#7ee787",
        warning: "#f2cc60",
        error: "#ff7b72",
    };
    ui.status.style.color = colors[kind] || colors.neutral;
}

function loadToken() {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) {
        ui.ionToken.value = saved;
    }
}

function setDefaultOriginInputs() {
    ui.originLat.value = defaultOrigin.latDeg.toFixed(6);
    ui.originLon.value = defaultOrigin.lonDeg.toFixed(6);
    ui.originAlt.value = defaultOrigin.altM.toFixed(1);
}

function currentOrigin() {
    return {
        latDeg: Number.parseFloat(ui.originLat.value),
        lonDeg: Number.parseFloat(ui.originLon.value),
        altM: Number.parseFloat(ui.originAlt.value),
    };
}

function validateOrigin(origin) {
    return Number.isFinite(origin.latDeg) && Number.isFinite(origin.lonDeg) && Number.isFinite(origin.altM);
}

function ensureViewer() {
    const token = ui.ionToken.value.trim();
    if (!token) {
        throw new Error("Paste a Cesium Ion token first.");
    }
    if (viewer) {
        return viewer;
    }

    Cesium.Ion.defaultAccessToken = token;
    viewer = new Cesium.Viewer("cesiumContainer", {
        terrain: Cesium.Terrain.fromWorldTerrain(),
        baseLayerPicker: true,
        timeline: true,
        animation: true,
        shouldAnimate: false,
        sceneModePicker: false,
        selectionIndicator: false,
        infoBox: false,
        shadows: true,
        terrainShadows: Cesium.ShadowMode.ENABLED,
        requestRenderMode: false,
        contextOptions: {
            webgl: { alpha: false, antialias: true, powerPreference: "high-performance" },
        },
    });

    viewer.scene.highDynamicRange = true;
    viewer.scene.globe.enableLighting = true;
    return viewer;
}

function parseCsv(text) {
    const lines = text.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (lines.length < 2) {
        throw new Error("CSV file is empty or missing data rows.");
    }

    const headers = splitCsvLine(lines[0]).map((item) => item.trim());
    const rows = [];
    for (let i = 1; i < lines.length; i += 1) {
        const values = splitCsvLine(lines[i]);
        if (values.length === 0) {
            continue;
        }
        const row = {};
        headers.forEach((header, index) => {
            row[header] = (values[index] ?? "").trim();
        });
        rows.push(row);
    }
    return rows;
}

function splitCsvLine(line) {
    const out = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i += 1) {
        const ch = line[i];
        if (ch === "\"") {
            if (inQuotes && line[i + 1] === "\"") {
                current += "\"";
                i += 1;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch === "," && !inQuotes) {
            out.push(current);
            current = "";
        } else {
            current += ch;
        }
    }
    out.push(current);
    return out;
}

function parseNumber(row, key) {
    const raw = row[key];
    if (raw === undefined || raw === "") {
        return NaN;
    }
    const value = Number.parseFloat(raw);
    return Number.isFinite(value) ? value : NaN;
}

function localOffsetsToGeodetic(eastM, northM, upM, origin) {
    const lat0Rad = origin.latDeg * DEG2RAD;
    const latDeg = origin.latDeg + (northM / EARTH_RADIUS_M) / DEG2RAD;
    const lonDeg = origin.lonDeg + (eastM / (EARTH_RADIUS_M * Math.cos(lat0Rad))) / DEG2RAD;
    const altM = origin.altM + upM;
    return { latDeg, lonDeg, altM };
}

function normalizeRows(rawRows, origin) {
    return rawRows.map((rawRow, idx) => {
        const timeS = parseNumber(rawRow, "time_s");
        const eastM = parseNumber(rawRow, "east_m");
        const northM = parseNumber(rawRow, "north_m");
        const upM = parseNumber(rawRow, "up_m");
        const hasLocal = Number.isFinite(eastM) && Number.isFinite(northM) && Number.isFinite(upM);

        let latDeg = parseNumber(rawRow, "lat_deg");
        let lonDeg = parseNumber(rawRow, "lon_deg");
        let altM = parseNumber(rawRow, "alt_m");
        if (hasLocal) {
            const geo = localOffsetsToGeodetic(eastM, northM, upM, origin);
            latDeg = geo.latDeg;
            lonDeg = geo.lonDeg;
            altM = geo.altM;
        }

        const headingDeg = parseNumber(rawRow, "heading_deg");
        const pitchDeg = parseNumber(rawRow, "pitch_deg");
        const rollDeg = parseNumber(rawRow, "roll_deg");
        const yawDeg = Number.isFinite(parseNumber(rawRow, "yaw_deg")) ? parseNumber(rawRow, "yaw_deg") : headingDeg - 90.0;

        return {
            index: idx,
            timeS,
            latDeg,
            lonDeg,
            altM,
            eastM,
            northM,
            upM,
            headingDeg,
            pitchDeg,
            rollDeg,
            yawDeg,
            uMps: parseNumber(rawRow, "u_mps"),
            vMps: parseNumber(rawRow, "v_mps"),
            wMps: parseNumber(rawRow, "w_mps"),
            pDegS: parseNumber(rawRow, "p_deg_s"),
            qDegS: parseNumber(rawRow, "q_deg_s"),
            rDegS: parseNumber(rawRow, "r_deg_s"),
            collectiveRpm: parseNumber(rawRow, "collective_rpm"),
            rpmLeft: parseNumber(rawRow, "rpm_left"),
            rpmRight: parseNumber(rawRow, "rpm_right"),
            motorRpms: Array.from({ length: 10 }, (_, motorIdx) => parseNumber(rawRow, `rpm_${motorIdx + 1}`)),
            elevatorDeg: parseNumber(rawRow, "elevator_deg"),
            aileronDeg: parseNumber(rawRow, "aileron_deg"),
            rudderDeg: parseNumber(rawRow, "rudder_deg"),
            flapDeg: parseNumber(rawRow, "flap_deg"),
            controller: rawRow.controller || "--",
            xM: parseNumber(rawRow, "x_m"),
            yM: parseNumber(rawRow, "y_m"),
            hM: parseNumber(rawRow, "h_m"),
        };
    }).filter((row) => (
        Number.isFinite(row.timeS)
        && Number.isFinite(row.latDeg)
        && Number.isFinite(row.lonDeg)
        && Number.isFinite(row.altM)
    ));
}

function clearElement(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}

function finiteOrZero(value) {
    return Number.isFinite(value) ? value : 0.0;
}

function computeCenteredScale(values, fallback) {
    let maxAbs = 0.0;
    values.forEach((value) => {
        if (Number.isFinite(value)) {
            maxAbs = Math.max(maxAbs, Math.abs(value));
        }
    });
    return Math.max(maxAbs, fallback);
}

function createBarWidget(label, kind, units) {
    const card = document.createElement("div");
    card.className = "control-card";

    const meta = document.createElement("div");
    meta.className = "control-meta";

    const labelEl = document.createElement("span");
    labelEl.className = "control-label";
    labelEl.textContent = label;

    const valueEl = document.createElement("span");
    valueEl.className = "control-value";
    valueEl.textContent = `0.0 ${units}`;

    meta.appendChild(labelEl);
    meta.appendChild(valueEl);

    const centerLabel = document.createElement("div");
    centerLabel.className = "baseline-label";
    centerLabel.textContent = units === "RPM" ? "-- RPM" : "-- deg";

    const track = document.createElement("div");
    track.className = "center-bar";

    const fill = document.createElement("div");
    fill.className = `bar-fill ${kind}`;
    track.appendChild(fill);

    card.appendChild(meta);
    card.appendChild(centerLabel);
    card.appendChild(track);

    return { card, valueEl, fill, centerLabel, units };
}

function updateBarWidget(widget, value, scale) {
    const safeValue = finiteOrZero(value);
    const safeScale = Math.max(scale, 1e-6);
    const normalized = Math.max(-1.0, Math.min(1.0, safeValue / safeScale));
    const magnitudePct = Math.abs(normalized) * 50.0;

    widget.valueEl.textContent = `${safeValue >= 0 ? "+" : ""}${formatNumber(safeValue, 1)} ${widget.units}`;
    widget.fill.style.width = `${magnitudePct}%`;
    widget.fill.style.left = normalized >= 0 ? "50%" : `${50.0 - magnitudePct}%`;
}

function setupControlHud(rows) {
    if (ui.sidebarMotorBars) {
        clearElement(ui.sidebarMotorBars);
    }
    if (ui.sidebarSurfaceBars) {
        clearElement(ui.sidebarSurfaceBars);
    }

    const baselineRow = rows.length ? rows[rows.length - 1] : null;
    const motorBaseline = baselineRow ? baselineRow.motorRpms.map((value) => finiteOrZero(value)) : [];
    const hasMotors = rows.some((row) => row.motorRpms.some((value) => Number.isFinite(value)));
    const sidebarMotorWidgets = [];
    let motorScale = 50.0;

    if (hasMotors) {
        const motorDeltas = [];
        rows.forEach((row) => {
            row.motorRpms.forEach((value, idx) => {
                if (Number.isFinite(value)) {
                    motorDeltas.push(value - motorBaseline[idx]);
                }
            });
        });
        motorScale = computeCenteredScale(motorDeltas, 50.0);
        for (let idx = 0; idx < 10; idx += 1) {
            if (ui.sidebarMotorBars) {
                const sidebarWidget = createBarWidget(`M${idx + 1}`, "motor", "RPM");
                sidebarWidget.centerLabel.textContent = `${formatNumber(motorBaseline[idx], 0)} RPM`;
                ui.sidebarMotorBars.appendChild(sidebarWidget.card);
                sidebarMotorWidgets.push(sidebarWidget);
            }
        }
        if (ui.sidebarMotorScaleText) {
            ui.sidebarMotorScaleText.textContent = `Motor scale: +/- ${formatNumber(motorScale, 0)} RPM`;
        }
    } else {
        if (ui.sidebarMotorScaleText) {
            ui.sidebarMotorScaleText.textContent = "Motor scale: unavailable";
        }
    }

    const surfaceDefs = [
        { key: "elevatorDeg", label: "Elev", units: "deg" },
        { key: "aileronDeg", label: "Ail", units: "deg" },
        { key: "rudderDeg", label: "Rud", units: "deg" },
        { key: "flapDeg", label: "Flap", units: "deg" },
    ].filter((def) => rows.some((row) => Number.isFinite(row[def.key])));

    const surfaceBaseline = Object.fromEntries(
        surfaceDefs.map((def) => [def.key, baselineRow ? finiteOrZero(baselineRow[def.key]) : 0.0]),
    );
    const surfaceDeltas = [];
    surfaceDefs.forEach((def) => {
        rows.forEach((row) => {
            if (Number.isFinite(row[def.key])) {
                surfaceDeltas.push(row[def.key] - surfaceBaseline[def.key]);
            }
        });
    });
    const surfaceScale = computeCenteredScale(surfaceDeltas, 2.0);
    const surfaceWidgets = surfaceDefs.map((def) => {
        let sidebarWidget = null;
        if (ui.sidebarSurfaceBars) {
            sidebarWidget = createBarWidget(def.label, "surface", def.units);
            sidebarWidget.centerLabel.textContent = `${formatNumber(surfaceBaseline[def.key], 1)} deg`;
            ui.sidebarSurfaceBars.appendChild(sidebarWidget.card);
        }
        return { ...def, sidebarWidget };
    });
    if (ui.sidebarSurfaceScaleText) {
        ui.sidebarSurfaceScaleText.textContent = `Surface scale: +/- ${formatNumber(surfaceScale, 1)} deg`;
    }

    const visible = hasMotors || surfaceWidgets.length > 0;
    activeControlHud = visible ? {
        motorBaseline,
        motorScale,
        sidebarMotorWidgets,
        surfaceBaseline,
        surfaceScale,
        surfaceWidgets,
    } : null;
}

function updateControlHud(row) {
    if (!activeControlHud) {
        return;
    }

    activeControlHud.sidebarMotorWidgets.forEach((widget, idx) => {
        if (activeControlHud.sidebarMotorWidgets[idx]) {
            const value = Number.isFinite(row.motorRpms[idx])
                ? row.motorRpms[idx] - activeControlHud.motorBaseline[idx]
                : 0.0;
            updateBarWidget(widget, value, activeControlHud.motorScale);
        }
    });

    activeControlHud.surfaceWidgets.forEach((def) => {
        const value = Number.isFinite(row[def.key])
            ? row[def.key] - activeControlHud.surfaceBaseline[def.key]
            : 0.0;
        if (def.sidebarWidget) {
            updateBarWidget(def.sidebarWidget, value, activeControlHud.surfaceScale);
        }
    });
}

function buildPlayback(rows) {
    const start = Cesium.JulianDate.now();
    const sampledPosition = new Cesium.SampledPositionProperty();
    const sampledOrientation = new Cesium.SampledProperty(Cesium.Quaternion);
    sampledPosition.setInterpolationOptions({
        interpolationDegree: 2,
        interpolationAlgorithm: Cesium.HermitePolynomialApproximation,
    });

    const modelAdjustment = Cesium.Quaternion.fromHeadingPitchRoll(
        new Cesium.HeadingPitchRoll(
            Cesium.Math.toRadians(MODEL_HEADING_CORRECTION_DEG),
            Cesium.Math.toRadians(MODEL_PITCH_CORRECTION_DEG),
            Cesium.Math.toRadians(MODEL_ROLL_CORRECTION_DEG),
        ),
    );

    let stop = start;
    rows.forEach((row) => {
        const time = Cesium.JulianDate.addSeconds(start, row.timeS, new Cesium.JulianDate());
        const position = Cesium.Cartesian3.fromDegrees(row.lonDeg, row.latDeg, row.altM);
        const baseQuat = Cesium.Transforms.headingPitchRollQuaternion(
            position,
            new Cesium.HeadingPitchRoll(
                Cesium.Math.toRadians(row.headingDeg),
                Cesium.Math.toRadians(row.rollDeg),
                Cesium.Math.toRadians(row.pitchDeg),
            ),
        );
        const finalQuat = Cesium.Quaternion.multiply(baseQuat, modelAdjustment, new Cesium.Quaternion());

        sampledPosition.addSample(time, position);
        sampledOrientation.addSample(time, finalQuat);
        stop = time;
    });

    return { start, stop, sampledPosition, sampledOrientation };
}

function propAxisVector() {
    if (PROP_ROTATION_AXIS === "y") {
        return Cesium.Cartesian3.UNIT_Y;
    }
    if (PROP_ROTATION_AXIS === "z") {
        return Cesium.Cartesian3.UNIT_Z;
    }
    return Cesium.Cartesian3.UNIT_X;
}

function createPropNodeTransformations(startTime) {
    const nodeTransformations = {};
    const axis = propAxisVector();

    PROP_METADATA.forEach((prop) => {
        nodeTransformations[prop.name] = new Cesium.CallbackProperty((time, result) => {
            const row = rowAtTime(time);
            const sourceRpm = prop.side === "left" ? row?.rpmLeft : row?.rpmRight;
            const rpm = Number.isFinite(sourceRpm) ? sourceRpm : 0.0;
            const visualRpm = rpm * PROP_VISUAL_RPM_SCALE;
            const elapsedS = Cesium.JulianDate.secondsDifference(time, startTime);
            const angleRad = prop.spinSign * elapsedS * visualRpm * 2.0 * Math.PI / 60.0;

            if (!result) {
                result = new Cesium.TranslationRotationScale();
                result.scale = new Cesium.Cartesian3(1.0, 1.0, 1.0);
            }
            result.translation = Cesium.Cartesian3.ZERO;
            result.rotation = Cesium.Quaternion.fromAxisAngle(axis, angleRad);
            return result;
        }, false);
    });

    return nodeTransformations;
}

function updateSummary(rows) {
    ui.rowCount.textContent = String(rows.length);
    const durationS = rows.length > 1 ? rows[rows.length - 1].timeS - rows[0].timeS : 0.0;
    ui.durationVal.textContent = formatNumber(durationS, 2);
    ui.controllerVal.textContent = rows.length ? rows[0].controller : "--";
}

function updateTelemetry(row) {
    ui.telemetryTime.textContent = `${formatNumber(row.timeS, 2)} s`;
    ui.telemetryLat.textContent = `${formatNumber(row.latDeg, 6)} deg`;
    ui.telemetryLon.textContent = `${formatNumber(row.lonDeg, 6)} deg`;
    ui.telemetryAlt.textContent = `${formatNumber(row.altM, 1)} m`;
    ui.telemetryRoll.textContent = `${formatNumber(row.rollDeg, 2)} deg`;
    ui.telemetryPitch.textContent = `${formatNumber(row.pitchDeg, 2)} deg`;
    ui.telemetryYaw.textContent = `${formatNumber(row.yawDeg, 2)} deg`;
    ui.telemetryUVW.textContent = `${formatNumber(row.uMps, 2)} / ${formatNumber(row.vMps, 2)} / ${formatNumber(row.wMps, 2)} m/s`;
    ui.telemetryPQR.textContent = `${formatNumber(row.pDegS, 2)} / ${formatNumber(row.qDegS, 2)} / ${formatNumber(row.rDegS, 2)} deg/s`;
    ui.telemetryRpmLR.textContent = `${formatNumber(row.rpmLeft, 1)} / ${formatNumber(row.rpmRight, 1)}`;
    ui.telemetrySurfaces.textContent = `${formatNumber(row.elevatorDeg, 2)} / ${formatNumber(row.aileronDeg, 2)} / ${formatNumber(row.rudderDeg, 2)} deg`;
    ui.hudText.textContent = [
        `${row.controller.toUpperCase()} playback`,
        `t=${formatNumber(row.timeS, 1)} s`,
        `hdg=${formatNumber(row.headingDeg, 1)} deg`,
    ].join(" | ");
    updateControlHud(row);
}

function rowAtTime(currentTime) {
    if (activeRows.length === 0) {
        return null;
    }
    const elapsed = Cesium.JulianDate.secondsDifference(currentTime, viewer.clock.startTime);
    if (elapsed <= activeRows[0].timeS) {
        return activeRows[0];
    }
    if (elapsed >= activeRows[activeRows.length - 1].timeS) {
        return activeRows[activeRows.length - 1];
    }

    let lo = 0;
    let hi = activeRows.length - 1;
    while (lo <= hi) {
        const mid = Math.floor((lo + hi) / 2);
        const timeMid = activeRows[mid].timeS;
        if (timeMid < elapsed) {
            lo = mid + 1;
        } else if (timeMid > elapsed) {
            hi = mid - 1;
        } else {
            return activeRows[mid];
        }
    }
    const i1 = Math.min(lo, activeRows.length - 1);
    const i0 = Math.max(i1 - 1, 0);
    return Math.abs(activeRows[i1].timeS - elapsed) < Math.abs(elapsed - activeRows[i0].timeS)
        ? activeRows[i1]
        : activeRows[i0];
}

function clearTickHandler() {
    if (viewer && tickHandler) {
        viewer.clock.onTick.removeEventListener(tickHandler);
        tickHandler = null;
    }
}

function visualizeRows(rows, origin) {
    const fgViewer = ensureViewer();
    clearTickHandler();
    fgViewer.entities.removeAll();
    activeRows = rows;
    activeOrigin = origin;
    setupControlHud(rows);

    const playback = buildPlayback(rows);
    activeEntity = fgViewer.entities.add({
        availability: new Cesium.TimeIntervalCollection([
            new Cesium.TimeInterval({
                start: playback.start,
                stop: playback.stop,
            }),
        ]),
        position: playback.sampledPosition,
        orientation: playback.sampledOrientation,
        model: {
            uri: "./static/models/blown.gltf",
            scale: 1.0,
            minimumPixelSize: 80,
            shadows: Cesium.ShadowMode.ENABLED,
            runAnimations: true,
            imageBasedLightingFactor: new Cesium.Cartesian2(1.5, 1.5),
            nodeTransformations: createPropNodeTransformations(playback.start),
        },
        path: {
            width: 3,
            leadTime: 0,
            trailTime: rows[rows.length - 1].timeS + 1.0,
            material: new Cesium.PolylineGlowMaterialProperty({
                glowPower: 0.2,
                color: Cesium.Color.CYAN,
            }),
        },
    });

    fgViewer.clock.startTime = playback.start.clone();
    fgViewer.clock.stopTime = playback.stop.clone();
    fgViewer.clock.currentTime = playback.start.clone();
    fgViewer.clock.multiplier = 1.0;
    fgViewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
    fgViewer.clock.shouldAnimate = true;

    tickHandler = (clock) => {
        const row = rowAtTime(clock.currentTime);
        if (row) {
            updateTelemetry(row);
        }
    };
    fgViewer.clock.onTick.addEventListener(tickHandler);
    updateTelemetry(rows[0]);
    updateSummary(rows);
    fgViewer.flyTo(activeEntity).then(() => {
        fgViewer.trackedEntity = activeEntity;
    });

    setStatus(
        `Loaded ${rows.length} rows from ${formatNumber(rows[0].latDeg, 5)}, ${formatNumber(rows[0].lonDeg, 5)} using ${rows[0].controller.toUpperCase()} playback.`,
        "success",
    );
}

async function handleVisualize() {
    try {
        const origin = currentOrigin();
        if (!validateOrigin(origin)) {
            throw new Error("Origin latitude, longitude, and altitude must all be valid numbers.");
        }
        if (ui.csvFile.files.length === 0) {
            throw new Error("Choose a flight-history CSV first.");
        }
        ensureViewer();

        const file = ui.csvFile.files[0];
        setStatus(`Parsing ${file.name}...`, "warning");
        const rawText = await file.text();
        const rawRows = parseCsv(rawText);
        const rows = normalizeRows(rawRows, origin);
        if (rows.length < 2) {
            throw new Error("The CSV did not contain enough usable state samples.");
        }
        visualizeRows(rows, origin);
    } catch (error) {
        setStatus(error.message || "Unable to visualize this flight history.", "error");
        console.error(error);
    }
}

function saveToken() {
    const token = ui.ionToken.value.trim();
    if (!token) {
        setStatus("Paste a Cesium Ion token before saving.", "warning");
        return;
    }
    window.localStorage.setItem(STORAGE_KEY, token);
    setStatus("Cesium token saved locally in this browser.", "success");
}

function initialize() {
    if (typeof Cesium === "undefined") {
        setStatus("Cesium failed to load. Make sure you are opening the page through the local server and not directly from the filesystem.", "error");
        return;
    }
    loadToken();
    setDefaultOriginInputs();
    setStatus("Cesium app loaded. Paste your token, pick a CSV, and press Visualize Flight.", "neutral");

    ui.saveTokenBtn.addEventListener("click", saveToken);
    ui.visualizeBtn.addEventListener("click", handleVisualize);
}

initialize();
