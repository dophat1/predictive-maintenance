from __future__ import annotations

import csv
import json
from pathlib import Path

from flask import Flask, jsonify, render_template_string, send_from_directory

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
app = Flask(__name__)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Metro Compressor | Data Explorer & Decision Report</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>
:root{--ink:#17211b;--lime:#dfff4f;--paper:#f3f1ea;--muted:#657067;--red:#e56e4d}*{box-sizing:border-box}body{margin:0;background:var(--paper);font:15px system-ui;color:var(--ink)}main{width:min(1160px,calc(100% - 32px));margin:auto;padding:58px 0}small{font-weight:700;letter-spacing:.12em;color:var(--muted)}h1{font-size:clamp(42px,7vw,78px);line-height:.95;letter-spacing:-.06em;max-width:900px;margin:18px 0}h2{letter-spacing:-.03em}p{color:var(--muted);line-height:1.65}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:35px 0}.metric,.panel,.story{background:white;border:1px solid #d4d2c9;padding:22px}.metric strong{display:block;font-size:33px}.metric span{font-size:12px;color:var(--muted)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}.panel h2{margin-top:0}.chart{height:330px}.note{margin:16px 0;padding:18px;background:#fff3cd;border-left:4px solid #dfac31}.story{margin:16px 0}.story h2{margin-top:0}.slider{width:100%;accent-color:#43583c}.controls{display:flex;gap:18px;align-items:center;margin:12px 0}.pill{background:#e9eee2;padding:7px 12px;border-radius:20px;font-size:12px}.rec{background:var(--ink);color:white;padding:24px;margin-top:20px}.rec p{color:#d2dad3}.matrix{width:100%;display:block}a{color:inherit}@media(max-width:750px){.metrics{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.chart{height:280px}}</style></head>
<body><main><small>METROPT-3 · DATA EXPLORER + DECISION REPORT</small><h1>Explore the compressor data. Understand the warning.</h1><p>Inspect real sensor behavior around reported incidents, then review what the predictive model can and cannot tell a maintenance team.</p>
<section class="panel"><h2>Explore sensor history</h2><p>Choose a sensor and time range. The dark line shows its 10-minute average; shaded regions mark documented failure periods. This reveals operating behavior and data context, not causal proof.</p><div class="controls"><label>Sensor <select id="sensor"></select></label><label>From <input id="dateStart" type="date" value="2020-04-01"></label><label>To <input id="dateEnd" type="date" value="2020-07-31"></label><label><input id="showStd" type="checkbox"> Show variability band</label></div><div id="sensorChart" class="chart" style="height:430px"></div><div id="sensorSummary" class="pill"></div></section>
<section class="grid"><article class="panel"><h2>What is in the data?</h2><div id="dataSummary"></div></article><article class="panel"><h2>Reported incidents</h2><div id="events" class="chart"></div></article></section>
<section class="story"><h2>What the data says in one minute</h2><p>The selected model flagged <b>{{ m.holdout.true_positive }} of {{ m.holdout.true_positive + m.holdout.false_negative }} labeled pre-failure windows</b> in the held-out July period. It also flagged <b>{{ m.holdout.false_positive }}</b> windows outside those labeled periods. So the current model is good at raising suspicion, but too noisy to trigger work orders automatically. One incident is far too little evidence to estimate expected savings.</p></section>
<section class="metrics"><div class="metric"><strong>{{ '%.1f%%'|format(m.holdout.recall*100) }}</strong><span>Pre-failure windows flagged</span></div><div class="metric"><strong>{{ '%.1f%%'|format(m.holdout.precision*100) }}</strong><span>Flagged windows in target period</span></div><div class="metric"><strong>{{ m.holdout.false_positive }}</strong><span>False-positive windows</span></div><div class="metric"><strong>{{ '%.3f'|format(m.holdout.pr_auc) }}</strong><span>PR-AUC · ranking quality</span></div></section>
<section class="story"><h2>What question did we evaluate?</h2><p>Every 10 minutes, the model sees a summary of recent sensor readings and estimates whether a reported air leak will begin in the next 24 hours. We removed readings during active failure periods. The model does not diagnose the fault, estimate remaining life, or prove that maintenance action would prevent a breakdown.</p><p>Evidence strength is <b>low</b>: only four failures were documented. Two earlier events were used for forward validation and the July event was held out. The result describes this single test incident, not a reliable fleet-wide success rate.</p></section>
<section class="grid"><article class="panel"><h2>Which model performed best?</h2><p>Models were ranked by average precision (PR-AUC) on future failure events. Higher is better; variability across only two validation events is shown.</p><div id="models" class="chart"></div></article><article class="panel"><h2>What did the holdout alerts look like?</h2><p>Each cell counts ten-minute windows, not unique maintenance jobs.</p><div id="confusion" class="chart"></div></article></section>
<section class="panel"><h2>Explore the alert trade-off</h2><p>Move the cutoff to see how many windows would be flagged. Lower cutoffs catch more target windows and create more alerts. This curve is descriptive for the held-out period; choosing a threshold from it would overfit the test.</p><div class="controls"><label for="threshold">Risk cutoff</label><input class="slider" id="threshold" type="range" min="0.02" max="0.80" step="0.01" value="{{ m.holdout.threshold }}"><span class="pill" id="cutoff"></span></div><div id="tradeoff" class="chart"></div><div id="decision" class="pill"></div></section>
<section class="grid"><article class="panel"><h2>What would a business do next?</h2><p>Use alerts as a review queue for an engineer, not as automatic work orders. First combine consecutive flagged windows into alert episodes; the current 5,027 flagged windows may represent far fewer distinct episodes. Log inspection findings and response actions. Then compare useful warnings and inspection workload with the existing maintenance process.</p></article><article class="panel"><h2>What do we need to estimate value?</h2><p>Downtime cost per hour, inspection cost, repair cost, response lead time, and a defined alert episode. With those inputs and more failure events, we can estimate avoided loss and unnecessary inspection burden. The present dataset cannot support a savings claim.</p></article></section>
<section class="rec"><h2>Recommendation</h2><p><b>Do not automate maintenance decisions from this model yet.</b> Run a shadow evaluation: group alerts into episodes, review them with maintenance engineers, and collect outcomes prospectively. Revisit a pilot only if alert precision and actionable lead time improve across additional failures.</p><p>Selected model: <b>{{ m.best_model }}</b>. It was selected because its mean validation PR-AUC was highest; its final holdout PR-AUC was only {{ '%.3f'|format(m.holdout.pr_auc) }}. A simple threshold of {{ '%.2f'|format(m.holdout.threshold) }} prioritizes recall, but the low precision demonstrates why threshold policy must reflect operational cost.</p></section>
<p>Dataset: <a href="https://archive.ics.uci.edu/dataset/791/metropt%2B3%2B">UCI MetroPT-3</a> · <a href="/api/metrics">Machine-readable results</a> · <a href="/artifacts/cv_results.csv">Download model comparison</a></p></main>
<script>const M={{m|tojson}}, C={{models|tojson}}, D={{data_summary|tojson}}, E={{events|tojson}}; const layout={margin:{l:55,r:20,t:20,b:55},paper_bgcolor:'white',plot_bgcolor:'white',font:{family:'system-ui',color:'#17211b'},xaxis:{gridcolor:'#e6e5df'},yaxis:{gridcolor:'#e6e5df'}};
document.getElementById('dataSummary').innerHTML='<p><b>'+D.rows.toLocaleString()+'</b> raw readings · '+D.start+' to '+D.end+'</p><p><b>'+D.window_rows.toLocaleString()+'</b> usable ten-minute windows · '+D.sensors.length+' analog sensors · '+E.length+' documented failures.</p><p>Data is one compressor, so it cannot establish fleet-wide reliability.</p>';
const selectors=document.getElementById('sensor'); D.sensors.forEach(s=>selectors.add(new Option(s.label,s.key))); 
Plotly.newPlot('events',[{type:'bar',x:E.map(e=>e.start),y:E.map(()=>1),width:E.map(e=>new Date(e.end)-new Date(e.start)),orientation:'v',marker:{color:'#df7553'},text:E.map(e=>e.name),hovertemplate:'%{text}<br>Start %{x}<extra></extra>'}],{...layout,yaxis:{visible:false},xaxis:{...layout.xaxis,type:'date',title:'Documented failure date'},showlegend:false},{responsive:true,displayModeBar:false});
async function drawSensor(){const s=selectors.value,a=document.getElementById('dateStart').value,b=document.getElementById('dateEnd').value,r=await fetch('/api/sensor?name='+encodeURIComponent(s)+'&start='+a+'&end='+b),d=await r.json();if(!r.ok){document.getElementById('sensorSummary').textContent=d.error;return}const traces=[{x:d.times,y:d.mean,mode:'lines',name:'10-min mean',line:{color:'#27392d',width:1.5},hovertemplate:'%{x}<br>%{y:.3f}<extra></extra>'}];if(document.getElementById('showStd').checked){traces.push({x:d.times,y:d.upper,mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'});traces.push({x:d.times,y:d.lower,mode:'lines',fill:'tonexty',fillcolor:'rgba(184,217,51,.22)',line:{width:0},name:'±1 std'});}for(const e of E){traces.push({x:[e.start,e.end],y:[d.min,d.min],mode:'lines',line:{color:'#df7553',width:7},name:e.name,hovertemplate:e.name+'<extra></extra>'})}Plotly.react('sensorChart',traces,{...layout,xaxis:{...layout.xaxis,type:'date',title:'Timestamp'},yaxis:{...layout.yaxis,title:d.unit},legend:{orientation:'h',y:1.12}},{responsive:true,displayModeBar:false});document.getElementById('sensorSummary').textContent=d.count.toLocaleString()+' windows · mean '+d.overall_mean.toFixed(3)+' '+d.unit+' · range '+d.min.toFixed(3)+' to '+d.max.toFixed(3)+' '+d.unit;}
['change','input'].forEach(ev=>{selectors.addEventListener(ev,drawSensor);document.getElementById('dateStart').addEventListener(ev,drawSensor);document.getElementById('dateEnd').addEventListener(ev,drawSensor);document.getElementById('showStd').addEventListener(ev,drawSensor)});drawSensor();
Plotly.newPlot('models',[{type:'bar',x:C.map(x=>x.pr_auc_mean),y:C.map(x=>x.model),orientation:'h',marker:{color:'#b8d933'},error_x:{type:'data',array:C.map(x=>x.pr_auc_std),visible:true},hovertemplate:'%{y}<br>Mean PR-AUC %{x:.3f}<extra></extra>'}],{...layout,xaxis:{...layout.xaxis,title:'Mean event-validation PR-AUC'},yaxis:{...layout.yaxis,autorange:'reversed'}},{responsive:true,displayModeBar:false});
Plotly.newPlot('confusion',[{type:'heatmap',z:[[M.holdout.true_negative,M.holdout.false_positive],[M.holdout.false_negative,M.holdout.true_positive]],x:['No alert','Alert'],y:['No failure soon','Failure within 24h'],colorscale:[[0,'#f3f1ea'],[1,'#b8d933']],showscale:false,texttemplate:'%{z:,}',hovertemplate:'Actual %{y}<br>Predicted %{x}<br>%{z:,} windows<extra></extra>'}],{...layout,yaxis:{...layout.yaxis,autorange:'reversed'}},{responsive:true,displayModeBar:false});
async function update(){const t=Number(document.getElementById('threshold').value),r=await fetch('/api/threshold?value='+t),d=await r.json();document.getElementById('cutoff').textContent='cutoff '+t.toFixed(2);document.getElementById('decision').textContent=d.alert_windows.toLocaleString()+' windows flagged · '+(d.recall*100).toFixed(1)+'% recall · '+(d.precision*100).toFixed(1)+'% precision';Plotly.react('tradeoff',[{x:d.curve.map(x=>x.alert_windows),y:d.curve.map(x=>x.recall),mode:'lines',name:'Recall',line:{color:'#17211b',width:3},hovertemplate:'%{x:,} flagged windows<br>%{y:.1%} recall<extra></extra>'},{x:d.curve.map(x=>x.alert_windows),y:d.curve.map(x=>x.precision),mode:'lines',name:'Precision',line:{color:'#d28742',width:3},hovertemplate:'%{x:,} flagged windows<br>%{y:.1%} precision<extra></extra>'},{x:[d.alert_windows],y:[d.recall],mode:'markers',name:'Selected cutoff',marker:{size:12,color:'#df5e43'},hovertemplate:'Selected cutoff<br>%{x:,} flagged<br>%{y:.1%} recall<extra></extra>'}],{...layout,xaxis:{...layout.xaxis,title:'Ten-minute windows flagged'},yaxis:{...layout.yaxis,title:'Rate',tickformat:'.0%',rangemode:'tozero'},legend:{orientation:'h',y:1.12}},{responsive:true,displayModeBar:false});}document.getElementById('threshold').addEventListener('input',update);update();</script></body></html>"""


def load_results() -> tuple[dict, list[dict]]:
    metrics_path = ARTIFACTS / "metrics.json"
    cv_path = ARTIFACTS / "cv_results.csv"
    if not metrics_path.exists() or not cv_path.exists():
        raise FileNotFoundError("Run `python train.py` before starting the dashboard.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with cv_path.open(encoding="utf-8", newline="") as handle:
        models = [{key: (float(value) if key != "model" else value) for key, value in row.items()} for row in csv.DictReader(handle)]
    return metrics, models


@app.get("/")
def dashboard():
    metrics, models = load_results()
    from src.features import ANALOG, FAILURES
    import pandas as pd
    raw = pd.read_csv(ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv", usecols=["timestamp", *ANALOG], parse_dates=["timestamp"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    data_summary = {"rows": len(raw), "start": raw.timestamp.min().strftime("%Y-%m-%d"), "end": raw.timestamp.max().strftime("%Y-%m-%d"), "window_rows": metrics["window_rows"],
                    "sensors": [{"key": c, "label": c.replace("_", " ")} for c in ANALOG]}
    events = [{"name": f"Incident {i}", "start": start, "end": end} for i, (start, end) in enumerate(FAILURES, 1)]
    return render_template_string(PAGE, m=metrics, models=models, data_summary=data_summary, events=events)


@app.get("/api/sensor")
def sensor_api():
    import pandas as pd
    from flask import request
    from src.features import ANALOG, FAILURES
    name = request.args.get("name", "TP2")
    if name not in ANALOG:
        return jsonify({"error": "Unknown sensor"}), 400
    start, end = request.args.get("start", "2020-02-01"), request.args.get("end", "2020-08-31")
    raw = pd.read_csv(ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv", usecols=["timestamp", name], parse_dates=["timestamp"])
    series = raw.set_index("timestamp")[name].resample("10min").agg(["mean", "std"]).dropna()
    series = series.loc[start:end]
    if series.empty:
        return jsonify({"error": "No sensor data in this date range"}), 404
    unit = "bar" if name in {"TP2", "TP3", "H1", "DV_pressure", "Reservoirs"} else ("°C" if name == "Oil_temperature" else "A")
    return jsonify({"times": series.index.strftime("%Y-%m-%dT%H:%M:%S").tolist(), "mean": series["mean"].fillna(0).tolist(),
                    "upper": (series["mean"] + series["std"].fillna(0)).tolist(), "lower": (series["mean"] - series["std"].fillna(0)).tolist(),
                    "count": len(series), "overall_mean": float(series["mean"].mean()), "min": float((series["mean"] - series["std"].fillna(0)).min()),
                    "max": float((series["mean"] + series["std"].fillna(0)).max()), "unit": unit})


@app.get("/api/metrics")
def metrics_api():
    metrics, _ = load_results()
    return jsonify(metrics)


@app.get("/api/threshold")
def threshold_api():
    import joblib
    import numpy as np
    import pandas as pd
    from src.data import load_dataset
    from src.features import make_features
    threshold = float(__import__("flask").request.args.get("value", "0.02"))
    threshold = min(0.99, max(0.001, threshold))
    metrics, _ = load_results()
    raw = load_dataset(ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv")
    X, y, _ = make_features(raw)
    test = X.index >= pd.Timestamp(metrics["development_end"])
    y_test = y.loc[test].to_numpy()
    model = joblib.load(ARTIFACTS / "best_model.joblib")
    probability = model.predict_proba(X.loc[test])[:, 1]
    curve = []
    for cutoff in np.linspace(0.02, 0.80, 40):
        pred = probability >= cutoff
        tp = int(np.sum(pred & (y_test == 1))); fp = int(np.sum(pred & (y_test == 0)))
        curve.append({"threshold": float(cutoff), "alert_windows": tp + fp, "recall": tp / max(1, int(y_test.sum())), "precision": tp / max(1, tp + fp)})
    pred = probability >= threshold
    tp = int(np.sum(pred & (y_test == 1))); fp = int(np.sum(pred & (y_test == 0)))
    return jsonify({"threshold": threshold, "alert_windows": tp + fp, "recall": tp / max(1, int(y_test.sum())), "precision": tp / max(1, tp + fp), "curve": curve})


@app.get("/artifacts/<path:name>")
def artifact(name: str):
    return send_from_directory(ARTIFACTS, name)


if __name__ == "__main__":
    app.run(debug=False, port=8501)
