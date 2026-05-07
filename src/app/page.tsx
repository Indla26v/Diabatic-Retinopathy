"use client";
import React, { useState, useRef, useCallback } from "react";
import {
  UploadCloud, AlertTriangle, Info, ChevronDown, Activity,
  Image as ImageIcon, X, Users, Shield, TrendingUp, Plus, Trash2, Eye
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"];

const RISK_CONFIG: Record<string, { color: string; bg: string; border: string; label: string; glow: string }> = {
  critical: { color: "#dc2626", bg: "rgba(220,38,38,0.08)", border: "rgba(220,38,38,0.25)", label: "Critical", glow: "0 0 20px rgba(220,38,38,0.15)" },
  high:     { color: "#ea580c", bg: "rgba(234,88,12,0.08)", border: "rgba(234,88,12,0.25)", label: "High",     glow: "0 0 20px rgba(234,88,12,0.15)" },
  elevated: { color: "#d97706", bg: "rgba(217,119,6,0.08)", border: "rgba(217,119,6,0.25)", label: "Elevated", glow: "0 0 20px rgba(217,119,6,0.15)" },
  moderate: { color: "#0d9488", bg: "rgba(13,148,136,0.08)", border: "rgba(13,148,136,0.25)", label: "Moderate", glow: "0 0 20px rgba(13,148,136,0.15)" },
  low:      { color: "#059669", bg: "rgba(5,150,105,0.08)", border: "rgba(5,150,105,0.25)", label: "Low",      glow: "0 0 20px rgba(5,150,105,0.15)" },
};

interface ScreenResult {
  index: number; filename: string; thumbnail: string;
  predicted_class: number; predicted_label: string;
  severity_score: number; risk_tier: string;
  probabilities: number[]; rank: number;
  gemini_explanation?: string;
}
interface ScreenResponse {
  total_images: number; screened: number; errors: number;
  model_val_auc: number; ranked_results: ScreenResult[];
  error_results: { index: number; filename: string; error: string }[];
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScreenResponse | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);
  const [showInfo, setShowInfo] = useState(false);
  const [expandedCard, setExpandedCard] = useState<number | null>(null);
  const [modalImage, setModalImage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const addFiles = useCallback((newFiles: FileList | File[]) => {
    const arr = Array.from(newFiles).filter(f => f.type.startsWith("image/"));
    if (arr.length === 0) return;
    const combined = [...files, ...arr].slice(0, 20);
    setFiles(combined);
    setPreviews(combined.map(f => URL.createObjectURL(f)));
    setResult(null);
  }, [files]);

  const removeFile = (idx: number) => {
    const next = files.filter((_, i) => i !== idx);
    setFiles(next);
    setPreviews(next.map(f => URL.createObjectURL(f)));
    setResult(null);
  };

  const handleDrag = (e: React.DragEvent) => { e.preventDefault(); setIsDragActive(e.type === "dragenter" || e.type === "dragover"); };
  const handleDrop = (e: React.DragEvent) => { e.preventDefault(); setIsDragActive(false); if (e.dataTransfer.files) addFiles(e.dataTransfer.files); };

  const runScreening = async () => {
    if (files.length === 0) return;
    setLoading(true);
    const formData = new FormData();
    files.forEach(f => formData.append("files", f));
    try {
      const res = await fetch("http://localhost:8000/screen", { method: "POST", body: formData });
      if (!res.ok) throw new Error("API error");
      const data: ScreenResponse = await res.json();
      setResult(data);
    } catch {
      alert("Screening failed. Ensure the backend is running on localhost:8000.");
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => { setFiles([]); setPreviews([]); setResult(null); setExpandedCard(null); };

  const severityBarWidth = (score: number) => `${Math.min((score / 4) * 100, 100)}%`;

  return (
    <div className="min-h-screen text-slate-800 font-sans tracking-tight selection:bg-teal-500 selection:text-white relative overflow-hidden">
      {/* Background */}
      <div className="fixed inset-0 z-[-1] bg-[#f8fafc] overflow-hidden">
        <motion.div animate={{ scale: [1, 1.2, 1], x: [0, 30, -20, 0], y: [0, -40, 20, 0] }} transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }} className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] rounded-full bg-teal-200/60 blur-[100px] mix-blend-multiply" />
        <motion.div animate={{ scale: [1, 1.1, 1], x: [0, -30, 20, 0], y: [0, 40, -20, 0] }} transition={{ duration: 18, repeat: Infinity, ease: "easeInOut", delay: 2 }} className="absolute top-[20%] right-[-10%] w-[60vw] h-[60vw] rounded-full bg-cyan-200/50 blur-[120px] mix-blend-multiply" />
        <motion.div animate={{ scale: [1, 1.3, 1], x: [0, 40, -30, 0], y: [0, -20, 40, 0] }} transition={{ duration: 20, repeat: Infinity, ease: "easeInOut", delay: 4 }} className="absolute bottom-[-10%] left-[10%] w-[50vw] h-[50vw] rounded-full bg-indigo-200/50 blur-[100px] mix-blend-multiply" />
        <div className="absolute inset-0 bg-white/30 backdrop-blur-[50px]" />
      </div>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 md:py-10">
        {/* Hero Header */}
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="text-center mb-8">
          <div className="inline-flex items-center gap-2 bg-teal-50/80 border border-teal-200/60 px-4 py-1.5 rounded-full mb-4">
            <Shield className="w-3.5 h-3.5 text-teal-600" />
            <span className="text-xs font-bold text-teal-700 tracking-wide">SCREENING MODE</span>
            <span className="text-[10px] font-bold bg-teal-600 text-white px-2 py-0.5 rounded-full">AUC 95.44%</span>
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold text-slate-900 tracking-tight">Patient Triage Queue</h2>
          <p className="text-sm text-slate-500 mt-2 max-w-lg mx-auto">Upload multiple retinal scans. The AI ranks patients by relative severity so clinicians know who to examine first.</p>
        </motion.div>

        <div className="grid lg:grid-cols-12 gap-8 items-start">
          {/* LEFT: Upload Panel */}
          <section className="lg:col-span-4 flex flex-col gap-5">
            <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-white/60 p-6">
              <h3 className="text-base font-bold text-slate-800 mb-4 flex items-center gap-2">
                <ImageIcon className="w-4.5 h-4.5 text-teal-600" /> Upload Scans
                <span className="ml-auto text-xs font-bold text-slate-400">{files.length}/20</span>
              </h3>

              <div
                className={`relative flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300 ${isDragActive ? "border-teal-400 bg-teal-50/50 scale-[1.01]" : "border-slate-300/60 bg-white/40 hover:bg-white/70 hover:border-slate-400"}`}
                onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag} onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input type="file" ref={fileInputRef} onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = ""; }} accept="image/*" multiple className="hidden" />
                <div className="w-12 h-12 bg-white/80 shadow-sm border border-white flex items-center justify-center rounded-xl mb-3">
                  <UploadCloud className="w-6 h-6 text-teal-500/80" />
                </div>
                <p className="text-sm font-semibold text-slate-700">Drop scans or click to browse</p>
                <p className="text-[11px] text-slate-400 mt-1">PNG / JPEG · Multiple files supported</p>
              </div>

              {/* File List */}
              {files.length > 0 && (
                <div className="mt-4 space-y-2 max-h-[280px] overflow-y-auto pr-1">
                  {files.map((f, i) => (
                    <div key={i} className="flex items-center gap-3 bg-white/50 border border-white/60 rounded-xl px-3 py-2 group">
                      <img src={previews[i]} alt="" className="w-10 h-10 rounded-lg object-cover border border-slate-200/50" />
                      <span className="text-xs font-semibold text-slate-700 truncate flex-1">{f.name}</span>
                      <button onClick={(e) => { e.stopPropagation(); removeFile(i); }} className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-red-500 transition-all p-1">
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {files.length > 0 && (
                <div className="flex gap-3 mt-5">
                  <button onClick={clearAll} className="flex-1 py-2.5 bg-white/60 border border-white shadow-sm text-slate-600 font-bold rounded-full hover:bg-white text-sm transition-all">Clear</button>
                  <button onClick={runScreening} disabled={loading || !!result} className="flex-[2] py-2.5 bg-teal-600 text-white font-bold rounded-full hover:bg-teal-500 hover:shadow-[0_8px_25px_rgb(20,184,166,0.3)] disabled:opacity-50 disabled:cursor-not-allowed shadow-[0_4px_15px_rgb(20,184,166,0.2)] text-sm transition-all">
                    {loading ? "Screening..." : result ? "Screening Complete" : `Screen ${files.length} Patient${files.length > 1 ? "s" : ""}`}
                  </button>
                </div>
              )}
            </div>

            {/* Info Panel */}
            <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-white/60 p-5">
              <button onClick={() => setShowInfo(!showInfo)} className="flex items-center justify-between w-full text-sm font-semibold text-slate-800 hover:text-teal-600 transition-colors">
                <div className="flex items-center gap-2"><Info className="w-4 h-4 text-slate-400" /> How It Works</div>
                <ChevronDown className={`w-4 h-4 transition-transform duration-300 ${showInfo ? "rotate-180" : ""}`} />
              </button>
              <AnimatePresence>
                {showInfo && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                    <div className="pt-4 space-y-3 text-xs text-slate-600">
                      <div className="bg-white/40 p-3 rounded-xl border border-white/60 space-y-2">
                        <p><strong className="text-slate-800">Screening vs Grading:</strong> Instead of predicting exact DR severity, the model <em>ranks</em> patients by relative severity — which it does with <strong>95.44% AUC accuracy</strong>.</p>
                        <p><strong className="text-slate-800">Severity Score:</strong> A continuous 0–4 score computed as the expected value of class probabilities. Higher = more severe.</p>
                        <p><strong className="text-slate-800">Why AUC matters:</strong> AUC measures the probability that the model ranks a more-severe patient higher than a less-severe one. Our 95.44% AUC means near-perfect patient ordering.</p>
                      </div>
                      <div className="bg-white/40 p-3 rounded-xl border border-white/60 space-y-1">
                        <p><strong>Model:</strong> EfficientNet-B3 + Transformer</p>
                        <p><strong>Ranking Accuracy (AUC):</strong> 95.44%</p>
                        <p><strong>Preprocessing:</strong> CLAHE via OpenCV</p>
                        <p><strong>Input:</strong> 224×224 tensor</p>
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </section>

          {/* RIGHT: Results Panel */}
          <section className="lg:col-span-8 flex flex-col gap-5">
            {!result && !loading && (
              <div className="flex-1 min-h-[450px] bg-white/30 backdrop-blur-xl rounded-3xl border border-white/50 flex flex-col items-center justify-center p-8 text-center shadow-[0_8px_30px_rgb(0,0,0,0.02)]">
                <div className="w-16 h-16 bg-white/60 rounded-full flex items-center justify-center mb-6 shadow-sm border border-white">
                  <Users className="w-8 h-8 text-slate-300" />
                </div>
                <h3 className="text-xl font-bold text-slate-700 tracking-tight">Triage Queue Empty</h3>
                <p className="text-sm text-slate-500 mt-2 max-w-sm">Upload retinal scans to generate a severity-ranked triage queue for clinical review.</p>
                <div className="mt-6 inline-flex flex-col text-[12px] text-slate-500 space-y-1.5 bg-white/40 p-4 rounded-2xl border border-white/60 text-left">
                  <p><strong className="text-slate-600">Model:</strong> EfficientNet-B3 + Transformer Hybrid</p>
                  <p><strong className="text-slate-600">Ranking Accuracy (Val AUC):</strong> 95.44%</p>
                  <p><strong className="text-slate-600">Use Case:</strong> Relative severity screening & triage</p>
                </div>
              </div>
            )}

            {loading && (
              <div className="flex-1 min-h-[450px] bg-white/60 backdrop-blur-xl rounded-3xl border border-white/60 flex flex-col items-center justify-center p-8 shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
                <div className="relative w-16 h-16 mb-6">
                  <div className="absolute inset-0 rounded-full border-4 border-slate-100 opacity-50" />
                  <div className="absolute inset-0 rounded-full border-4 border-teal-500 border-t-transparent animate-spin" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 tracking-tight mb-2">Screening {files.length} Scan{files.length > 1 ? "s" : ""}...</h3>
                <p className="text-sm text-slate-500">Running CLAHE preprocessing & model inference</p>
              </div>
            )}

            <AnimatePresence>
              {result && !loading && (
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }} className="flex flex-col gap-5">
                  {/* Summary Bar */}
                  <div className="bg-white/60 backdrop-blur-xl rounded-2xl border border-white/60 p-5 shadow-[0_8px_30px_rgb(0,0,0,0.04)] flex flex-wrap items-center gap-6">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-teal-50 border border-teal-200/50 flex items-center justify-center"><TrendingUp className="w-5 h-5 text-teal-600" /></div>
                      <div>
                        <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Screened</p>
                        <p className="text-lg font-extrabold text-slate-900">{result.screened} <span className="text-sm text-slate-400 font-semibold">patient{result.screened > 1 ? "s" : ""}</span></p>
                      </div>
                    </div>
                    <div className="h-8 w-px bg-slate-200/60 hidden sm:block" />
                    <div>
                      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Model Ranking Accuracy</p>
                      <p className="text-lg font-extrabold text-teal-600">{(result.model_val_auc * 100).toFixed(2)}% <span className="text-sm text-slate-400 font-semibold">Val AUC</span></p>
                    </div>
                    <div className="h-8 w-px bg-slate-200/60 hidden sm:block" />
                    <div>
                      <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Queue Order</p>
                      <p className="text-sm font-bold text-slate-700">Most → Least Severe</p>
                    </div>
                    <button onClick={clearAll} className="ml-auto text-xs font-bold text-slate-500 hover:text-teal-600 bg-white/60 border border-white rounded-full px-4 py-2 transition-all hover:shadow-sm">New Screening</button>
                  </div>

                  {/* Ranked Cards */}
                  <div className="space-y-3">
                    {result.ranked_results.map((patient, i) => {
                      const rc = RISK_CONFIG[patient.risk_tier] || RISK_CONFIG.low;
                      const isExpanded = expandedCard === i;
                      return (
                        <motion.div
                          key={patient.filename + patient.index}
                          initial={{ opacity: 0, y: 15 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.08, duration: 0.4 }}
                          className="bg-white/60 backdrop-blur-xl rounded-2xl border border-white/60 shadow-[0_4px_20px_rgb(0,0,0,0.03)] overflow-hidden transition-all duration-300 hover:shadow-[0_8px_30px_rgb(0,0,0,0.06)]"
                          style={{ borderLeft: `3px solid ${rc.color}` }}
                        >
                          <div className="flex items-center gap-4 p-4 sm:p-5 cursor-pointer" onClick={() => setExpandedCard(isExpanded ? null : i)}>
                            {/* Rank Badge */}
                            <div className="w-10 h-10 rounded-xl flex items-center justify-center text-lg font-extrabold shrink-0" style={{ backgroundColor: rc.bg, color: rc.color, boxShadow: rc.glow }}>
                              {patient.rank}
                            </div>

                            {/* Thumbnail */}
                            <div className="w-12 h-12 rounded-xl overflow-hidden border border-slate-200/50 shrink-0 cursor-zoom-in" onClick={(e) => { e.stopPropagation(); setModalImage(patient.thumbnail); }}>
                              <img src={patient.thumbnail} alt="" className="w-full h-full object-cover" />
                            </div>

                            {/* Info */}
                            <div className="flex-1 min-w-0">
                              <p className="text-sm font-bold text-slate-800 truncate">{patient.filename}</p>
                              <div className="flex items-center gap-2 mt-1">
                                <span className="text-[10px] font-bold px-2 py-0.5 rounded-full" style={{ backgroundColor: rc.bg, color: rc.color, border: `1px solid ${rc.border}` }}>
                                  {rc.label} Risk
                                </span>
                                <span className="text-[11px] text-slate-500 font-medium">{patient.predicted_label}</span>
                              </div>
                            </div>

                            {/* Severity Score */}
                            <div className="text-right shrink-0 hidden sm:block">
                              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Severity Score</p>
                              <p className="text-2xl font-extrabold tracking-tight" style={{ color: rc.color }}>{patient.severity_score.toFixed(2)}<span className="text-xs text-slate-400 font-semibold ml-0.5">/4</span></p>
                            </div>

                            {/* Severity Bar */}
                            <div className="w-24 shrink-0 hidden md:block">
                              <div className="h-2.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                <motion.div initial={{ width: 0 }} animate={{ width: severityBarWidth(patient.severity_score) }} transition={{ duration: 0.8, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }} className="h-full rounded-full" style={{ backgroundColor: rc.color }} />
                              </div>
                            </div>

                            <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform duration-300 shrink-0 ${isExpanded ? "rotate-180" : ""}`} />
                          </div>

                          {/* Expanded Details */}
                          <AnimatePresence>
                            {isExpanded && (
                              <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                                <div className="px-5 pb-5 pt-1 border-t border-slate-100/60">
                                  {/* Mobile severity score */}
                                  <div className="sm:hidden mb-4 flex items-center justify-between">
                                    <span className="text-xs font-bold text-slate-500">Severity Score</span>
                                    <span className="text-xl font-extrabold" style={{ color: rc.color }}>{patient.severity_score.toFixed(2)}/4</span>
                                  </div>
                                  <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest mb-3">Class Probability Distribution</p>
                                  <div className="space-y-2.5">
                                    {patient.probabilities.map((prob, ci) => (
                                      <div key={ci}>
                                        <div className="flex justify-between text-xs font-semibold mb-1">
                                          <span className="text-slate-600">{CLASS_NAMES[ci]}</span>
                                          <span className="text-slate-500">{(prob * 100).toFixed(1)}%</span>
                                        </div>
                                        <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                                          <motion.div initial={{ width: 0 }} animate={{ width: `${Math.max(prob * 100, 1)}%` }} transition={{ duration: 0.6, delay: ci * 0.05 }} className="h-full rounded-full opacity-70" style={{ backgroundColor: rc.color }} />
                                        </div>
                                      </div>
                                    ))}
                                  </div>
                                  
                                  {/* Gemini AI explanation */}
                                  {patient.gemini_explanation && (
                                    <div className="mt-5 p-4 bg-teal-50/50 rounded-xl border border-teal-100/60">
                                      <div className="flex items-center gap-2 mb-2">
                                        <Activity className="w-4 h-4 text-teal-600" />
                                        <span className="text-xs font-bold text-teal-800 uppercase tracking-wider">AI Ophthalmologist Assessment</span>
                                      </div>
                                      <p className="text-sm text-slate-700 leading-relaxed whitespace-pre-wrap">
                                        {patient.gemini_explanation}
                                      </p>
                                    </div>
                                  )}
                                </div>
                              </motion.div>
                            )}
                          </AnimatePresence>
                        </motion.div>
                      );
                    })}
                  </div>

                  {/* Clinical Advisory */}
                  <div className="bg-orange-50/50 border border-orange-200/30 p-5 rounded-2xl backdrop-blur-md">
                    <div className="flex items-start gap-3">
                      <div className="bg-orange-100 p-2 rounded-xl text-orange-600 shrink-0"><AlertTriangle className="w-4 h-4" /></div>
                      <div>
                        <h4 className="text-sm font-bold text-orange-900 tracking-tight mb-1">Clinical Advisory</h4>
                        <p className="text-[12px] text-orange-800/80 leading-relaxed">
                          This AI screening ranks patients by <strong>relative</strong> severity (Val AUC: 95.44%). It does not replace a comprehensive ophthalmic exam. The ranking order is statistically reliable, but individual class predictions should be verified by a specialist.
                        </p>
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </section>
        </div>
      </main>

      {/* Image Modal */}
      <AnimatePresence>
        {modalImage && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={() => setModalImage(null)}>
            <button className="absolute top-4 right-4 z-[100] bg-black/60 text-white rounded-full p-2 backdrop-blur-md hover:bg-black/80 shadow-2xl" onClick={() => setModalImage(null)}>
              <X className="w-6 h-6" />
            </button>
            <motion.img initial={{ scale: 0.9 }} animate={{ scale: 1 }} exit={{ scale: 0.9 }} src={modalImage} alt="Full view" className="max-w-4xl max-h-[85vh] rounded-2xl shadow-2xl object-contain" onClick={e => e.stopPropagation()} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
