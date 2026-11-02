import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field, asdict

@dataclass
class ClientTrainEvent:
    round_num: Any
    client_id: Any
    timestamp: Any
    loss: Any
    num_samples: Any
    num_batches: Any
    custom_metrics: Any = field(default_factory=dict)
    debug_info: Any = None

@dataclass
class AggregationEvent:
    round_num: Any
    timestamp: Any
    num_clients: Any
    total_samples: Any
    aggregation_method: Any
    custom_metrics: Any = field(default_factory=dict)
    debug_info: Any = None

@dataclass
class DebugEvent:
    round_num: Any
    timestamp: Any
    event_type: Any
    severity: Any
    message: Any
    data: Any = field(default_factory=dict)

@dataclass
class GradientEvent:
    round_num: Any
    client_id: Any
    timestamp: Any
    grad_a_norm: Any
    grad_b_norm: Any
    grad_m_norm: Any
    grad_cls_norm: Any
    grad_ratio_a_over_b: Any
    per_layer: Any = None

@dataclass
class ParamNormEvent:
    round_num: Any
    timestamp: Any
    phase: Any
    m_norms: Any = field(default_factory=dict)
    a_norms: Any = field(default_factory=dict)
    b_norms: Any = field(default_factory=dict)
    m_total: Any = 0.0
    a_total: Any = 0.0
    b_total: Any = 0.0

@dataclass
class LRStateEvent:
    round_num: Any
    timestamp: Any
    lr_factor: Any
    effective_lrs: Any = field(default_factory=dict)

@dataclass
class RoundSummary:
    round_num: Any
    timestamp: Any
    train_loss: Any
    val_accuracy: Any = None
    lr_factor: Any = 1.0
    m_similarity: Any = None
    a_similarity: Any = None
    b_similarity: Any = None
    similarity_ordering: Any = None
    avg_grad_a_norm: Any = None
    avg_grad_b_norm: Any = None
    avg_grad_ratio: Any = None
    b_norm_mean: Any = None
    b_norm_cv: Any = None

class MethodLogger:

    def __init__(self, method_name, output_dir, log_level='INFO', enable_json=True, enable_traces=True):
        self.method_name = method_name
        self.output_dir = Path(output_dir)
        self.enable_json = enable_json
        self.enable_traces = enable_traces
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if enable_traces:
            self.traces_dir = self.output_dir / 'traces'
            self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f'fedfl.{method_name}')
        level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(level)
        method_log_path = self.logs_dir / f'{method_name}.log'
        already_has_handler = any((isinstance(h, logging.FileHandler) and h.baseFilename == str(method_log_path.resolve()) for h in self.logger.handlers))
        if not already_has_handler:
            fh = logging.FileHandler(method_log_path, mode='a')
            fh.setLevel(level)
            formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        self.client_events = []
        self.aggregation_events = []
        self.debug_events = []
        self.gradient_events = []
        self.param_norm_events = []
        self.lr_state_events = []
        self.round_summaries = []
        self.round_traces = {}
        self._last_val_acc = None
        self.logger.info(f'MethodLogger initialized for {method_name}')

    def log_client_train(self, client_id, round_num, loss, num_samples, num_batches, custom_metrics=None, debug_info=None):
        event = ClientTrainEvent(round_num=round_num, client_id=client_id, timestamp=datetime.now().isoformat(), loss=loss, num_samples=num_samples, num_batches=num_batches, custom_metrics=custom_metrics or {}, debug_info=debug_info)
        self.client_events.append(event)
        metrics_str = ' | '.join((f'{k}={v:.4f}' for (k, v) in (custom_metrics or {}).items()))
        self.logger.debug(f'R{round_num:03d} C{client_id:03d} | loss={loss:.4f} | samples={num_samples} | {metrics_str}')
        if self.enable_traces:
            if round_num not in self.round_traces:
                self.round_traces[round_num] = {'clients': [], 'aggregation': None}
            self.round_traces[round_num]['clients'].append(asdict(event))

    def log_aggregation(self, round_num, num_clients, total_samples, aggregation_method='weighted_avg', custom_metrics=None, debug_info=None):
        event = AggregationEvent(round_num=round_num, timestamp=datetime.now().isoformat(), num_clients=num_clients, total_samples=total_samples, aggregation_method=aggregation_method, custom_metrics=custom_metrics or {}, debug_info=debug_info)
        self.aggregation_events.append(event)
        metrics_str = ' | '.join((f'{k}={v:.4f}' for (k, v) in (custom_metrics or {}).items()))
        self.logger.info(f'R{round_num:03d} AGG | clients={num_clients} | samples={total_samples} | method={aggregation_method} | {metrics_str}')
        if self.enable_traces:
            if round_num not in self.round_traces:
                self.round_traces[round_num] = {'clients': [], 'aggregation': None}
            self.round_traces[round_num]['aggregation'] = asdict(event)

    def log_debug_event(self, event_type, round_num, message, data=None, severity='info'):
        event = DebugEvent(round_num=round_num, timestamp=datetime.now().isoformat(), event_type=event_type, severity=severity, message=message, data=data or {})
        self.debug_events.append(event)
        log_func = getattr(self.logger, severity, self.logger.info)
        log_func(f'R{round_num:03d} [{event_type}] {message}')

    def log_drift_metrics(self, round_num, magnitude_drift, direction_drift, client_id=None):
        prefix = f'C{client_id:03d}' if client_id is not None else 'GLOBAL'
        self.log_debug_event(event_type='drift_metrics', round_num=round_num, message=f'{prefix} m_drift={magnitude_drift:.6f} d_drift={direction_drift:.6f}', data={'magnitude_drift': magnitude_drift, 'direction_drift': direction_drift, 'client_id': client_id})

    def log_convergence_metrics(self, round_num, accuracy, loss, delta_accuracy=None):
        data = {'accuracy': accuracy, 'loss': loss}
        if delta_accuracy is not None:
            data['delta_accuracy'] = delta_accuracy
        severity = 'warning' if delta_accuracy and delta_accuracy < -0.01 else 'info'
        self.log_debug_event(event_type='convergence', round_num=round_num, message=f'acc={accuracy:.4f} loss={loss:.4f}', data=data, severity=severity)

    def log_custom(self, event_type, round_num, **kwargs):
        msg_parts = []
        for (k, v) in kwargs.items():
            if isinstance(v, float):
                msg_parts.append(f'{k}={v:.4f}')
            elif isinstance(v, (int, str)):
                msg_parts.append(f'{k}={v}')
            elif isinstance(v, dict) and len(v) <= 3:
                dict_str = ', '.join((f'{dk}={dv:.3f}' if isinstance(dv, float) else f'{dk}={dv}' for (dk, dv) in list(v.items())[:3]))
                msg_parts.append(f'{k}={{{dict_str}}}')
        message = ' | '.join(msg_parts) if msg_parts else f'{event_type} data logged'
        self.log_debug_event(event_type=event_type, round_num=round_num, message=message, data=kwargs, severity='info')

    def log_gradient_stats(self, round_num, client_id, grad_a_norm, grad_b_norm, grad_m_norm=0.0, grad_cls_norm=0.0, per_layer=None):
        ratio = grad_a_norm / max(grad_b_norm, 1e-12)
        event = GradientEvent(round_num=round_num, client_id=client_id, timestamp=datetime.now().isoformat(), grad_a_norm=grad_a_norm, grad_b_norm=grad_b_norm, grad_m_norm=grad_m_norm, grad_cls_norm=grad_cls_norm, grad_ratio_a_over_b=ratio, per_layer=per_layer)
        self.gradient_events.append(event)
        self.logger.debug(f'R{round_num:03d} C{client_id:03d} | grad A={grad_a_norm:.6f} B={grad_b_norm:.6f} m={grad_m_norm:.6f} cls={grad_cls_norm:.6f} ratio={ratio:.4f}')

    def log_param_norms(self, round_num, phase, m_norms=None, a_norms=None, b_norms=None):
        m_norms = m_norms or {}
        a_norms = a_norms or {}
        b_norms = b_norms or {}
        event = ParamNormEvent(round_num=round_num, timestamp=datetime.now().isoformat(), phase=phase, m_norms=m_norms, a_norms=a_norms, b_norms=b_norms, m_total=sum(m_norms.values()) if m_norms else 0.0, a_total=sum(a_norms.values()) if a_norms else 0.0, b_total=sum(b_norms.values()) if b_norms else 0.0)
        self.param_norm_events.append(event)
        self.logger.debug(f'R{round_num:03d} [{phase}] norms: m={event.m_total:.4f} A={event.a_total:.4f} B={event.b_total:.4f}')

    def log_lr_state(self, round_num, lr_factor, effective_lrs=None):
        event = LRStateEvent(round_num=round_num, timestamp=datetime.now().isoformat(), lr_factor=lr_factor, effective_lrs=effective_lrs or {})
        self.lr_state_events.append(event)
        self.logger.debug(f'R{round_num:03d} lr_factor={lr_factor:.6f}')

    def log_round_summary(self, round_num, train_loss, val_accuracy=None, lr_factor=1.0, m_similarity=None, a_similarity=None, b_similarity=None, similarity_ordering=None, avg_grad_a_norm=None, avg_grad_b_norm=None, avg_grad_ratio=None, b_norm_mean=None, b_norm_cv=None):
        summary = RoundSummary(round_num=round_num, timestamp=datetime.now().isoformat(), train_loss=train_loss, val_accuracy=val_accuracy, lr_factor=lr_factor, m_similarity=m_similarity, a_similarity=a_similarity, b_similarity=b_similarity, similarity_ordering=similarity_ordering, avg_grad_a_norm=avg_grad_a_norm, avg_grad_b_norm=avg_grad_b_norm, avg_grad_ratio=avg_grad_ratio, b_norm_mean=b_norm_mean, b_norm_cv=b_norm_cv)
        self.round_summaries.append(summary)
        self.logger.info(f"R{round_num:03d} SUMMARY | loss={train_loss:.4f} acc={(val_accuracy if val_accuracy is not None else 'N/A')} ordering={similarity_ordering or 'N/A'}")

    def log_aggregate_analysis(self, round_num, global_model, client_updates, lr_factor=1.0, effective_lrs=None, similarity_metrics=None, has_magnitude=False):
        import torch
        m_norms = {}
        a_norms = {}
        b_norms = {}
        for (name, param) in global_model.named_parameters():
            if not param.requires_grad:
                continue
            norm_val = torch.norm(param.float(), p='fro').item()
            short = name.split('.')[-2] + '.' + name.split('.')[-1] if '.' in name else name
            if has_magnitude and 'lora_magnitude_vector' in name:
                m_norms[short] = norm_val
            elif 'lora_A' in name:
                a_norms[short] = norm_val
            elif 'lora_B' in name:
                b_norms[short] = norm_val
        self.log_param_norms(round_num=round_num, phase='post_agg', m_norms=m_norms, a_norms=a_norms, b_norms=b_norms)
        self.log_lr_state(round_num=round_num, lr_factor=lr_factor, effective_lrs=effective_lrs or {})
        avg_loss = sum((u.loss for u in client_updates)) / len(client_updates)
        round_grads = [g for g in self.gradient_events if g.round_num == round_num]
        avg_ga = sum((g.grad_a_norm for g in round_grads)) / len(round_grads) if round_grads else None
        avg_gb = sum((g.grad_b_norm for g in round_grads)) / len(round_grads) if round_grads else None
        avg_gr = sum((g.grad_ratio_a_over_b for g in round_grads)) / len(round_grads) if round_grads else None
        b_nm = None
        b_cv_val = None
        if len(client_updates) >= 2:
            b_norms_list = []
            for u in client_updates:
                cb = 0.0
                bc = 0
                for (pn, pv) in u.state_dict.items():
                    if 'lora_B' in pn or 'lora_b' in pn:
                        cb += torch.norm(pv.float(), p='fro').item()
                        bc += 1
                if bc > 0:
                    b_norms_list.append(cb / bc)
            if b_norms_list:
                b_nm = sum(b_norms_list) / len(b_norms_list)
                b_std = (sum(((x - b_nm) ** 2 for x in b_norms_list)) / len(b_norms_list)) ** 0.5
                b_cv_val = b_std / (b_nm + 1e-08)
        m_sim = a_sim = b_sim = None
        sim_ordering = None
        if similarity_metrics is not None and hasattr(similarity_metrics, 'a_similarity_mean') and (len(client_updates) >= 2):
            a_sim = similarity_metrics.a_similarity_mean
            b_sim = similarity_metrics.b_similarity_mean
            if hasattr(similarity_metrics, 'm_similarity_mean'):
                m_sim = similarity_metrics.m_similarity_mean
            if hasattr(similarity_metrics, 'get_ordering'):
                sim_ordering = similarity_metrics.get_ordering()
        self.log_round_summary(round_num=round_num, train_loss=avg_loss, lr_factor=lr_factor, m_similarity=m_sim, a_similarity=a_sim, b_similarity=b_sim, similarity_ordering=sim_ordering, avg_grad_a_norm=avg_ga, avg_grad_b_norm=avg_gb, avg_grad_ratio=avg_gr, b_norm_mean=b_nm, b_norm_cv=b_cv_val)

    def get_experiment_report(self):
        if not self.round_summaries:
            return {}
        summaries = sorted(self.round_summaries, key=lambda s: s.round_num)
        n = len(summaries)
        losses = [s.train_loss for s in summaries]
        accs = [s.val_accuracy for s in summaries if s.val_accuracy is not None]
        pct10 = max(1, n // 10)
        early_loss = sum(losses[:pct10]) / pct10
        late_loss = sum(losses[-pct10:]) / pct10
        if late_loss < early_loss * 0.9:
            loss_trend = 'decreasing'
        elif late_loss > early_loss * 1.1:
            loss_trend = 'diverging'
        else:
            loss_trend = 'stalled'
        best_acc = max(accs) if accs else None
        best_acc_round = None
        if accs:
            for s in summaries:
                if s.val_accuracy == best_acc:
                    best_acc_round = s.round_num
                    break
        convergence = {'final_loss': losses[-1], 'best_accuracy': best_acc, 'best_round': best_acc_round, 'loss_trend': loss_trend, 'early_loss': early_loss, 'late_loss': late_loss}
        grad_ratios = [s.avg_grad_ratio for s in summaries if s.avg_grad_ratio is not None]
        gradient_flow = {}
        anomalies = []
        if grad_ratios:
            avg_ratio = sum(grad_ratios) / len(grad_ratios)
            early_r = sum(grad_ratios[:pct10]) / max(1, min(pct10, len(grad_ratios)))
            late_r = sum(grad_ratios[-pct10:]) / max(1, min(pct10, len(grad_ratios)))
            if late_r > early_r * 1.2:
                ratio_trend = 'increasing'
            elif late_r < early_r * 0.8:
                ratio_trend = 'decreasing'
            else:
                ratio_trend = 'stable'
            a_norms = [s.avg_grad_a_norm for s in summaries if s.avg_grad_a_norm is not None]
            b_norms = [s.avg_grad_b_norm for s in summaries if s.avg_grad_b_norm is not None]
            mean_a = sum(a_norms) / len(a_norms) if a_norms else 0.0
            mean_b = sum(b_norms) / len(b_norms) if b_norms else 0.0
            for s in summaries:
                if s.avg_grad_a_norm is not None and mean_a > 0 and (s.avg_grad_a_norm > mean_a * 5):
                    anomalies.append({'round': s.round_num, 'type': 'grad_spike_a', 'value': s.avg_grad_a_norm})
                if s.avg_grad_b_norm is not None and mean_b > 0 and (s.avg_grad_b_norm > mean_b * 5):
                    anomalies.append({'round': s.round_num, 'type': 'grad_spike_b', 'value': s.avg_grad_b_norm})
            gradient_flow = {'avg_a_over_b_ratio': avg_ratio, 'ratio_trend': ratio_trend, 'anomalies': anomalies}
        orderings = [s.similarity_ordering for s in summaries if s.similarity_ordering]
        similarity_evolution = {}
        if orderings:
            counter = Counter(orderings)
            dominant = counter.most_common(1)[0][0]
            stability = counter[dominant] / len(orderings)
            m_sims = [s.m_similarity for s in summaries if s.m_similarity is not None]
            a_sims = [s.a_similarity for s in summaries if s.a_similarity is not None]
            b_sims = [s.b_similarity for s in summaries if s.b_similarity is not None]
            similarity_evolution = {'dominant_ordering': dominant, 'ordering_stability': round(stability, 3), 'early_vs_late': {'early_m': sum(m_sims[:pct10]) / max(1, min(pct10, len(m_sims))) if m_sims else None, 'late_m': sum(m_sims[-pct10:]) / max(1, min(pct10, len(m_sims))) if m_sims else None, 'early_a': sum(a_sims[:pct10]) / max(1, min(pct10, len(a_sims))) if a_sims else None, 'late_a': sum(a_sims[-pct10:]) / max(1, min(pct10, len(a_sims))) if a_sims else None, 'early_b': sum(b_sims[:pct10]) / max(1, min(pct10, len(b_sims))) if b_sims else None, 'late_b': sum(b_sims[-pct10:]) / max(1, min(pct10, len(b_sims))) if b_sims else None}}
        b_means = [s.b_norm_mean for s in summaries if s.b_norm_mean is not None]
        b_cvs = [s.b_norm_cv for s in summaries if s.b_norm_cv is not None]
        param_norms = {}
        if b_means:
            early_b = sum(b_means[:pct10]) / max(1, min(pct10, len(b_means)))
            late_b = sum(b_means[-pct10:]) / max(1, min(pct10, len(b_means)))
            if late_b > early_b * 1.1:
                b_trend = 'growing'
            elif late_b < early_b * 0.9:
                b_trend = 'shrinking'
            else:
                b_trend = 'stable'
            param_norms = {'b_norm_trend': b_trend, 'b_cv_trend': {'early': sum(b_cvs[:pct10]) / max(1, min(pct10, len(b_cvs))) if b_cvs else None, 'late': sum(b_cvs[-pct10:]) / max(1, min(pct10, len(b_cvs))) if b_cvs else None}}
        gradient_explosions = len([a for a in anomalies if 'spike' in a.get('type', '')])
        loss_spikes = 0
        for i in range(1, len(losses)):
            if losses[i] > losses[i - 1] * 1.5:
                loss_spikes += 1
        similarity_drops = 0
        b_sims_all = [s.b_similarity for s in summaries if s.b_similarity is not None]
        for i in range(1, len(b_sims_all)):
            if b_sims_all[i] < b_sims_all[i - 1] * 0.8:
                similarity_drops += 1
        issues = gradient_explosions + loss_spikes + similarity_drops
        if issues == 0:
            overall = 'healthy'
        elif issues <= 3:
            overall = 'warning'
        else:
            overall = 'critical'
        health = {'gradient_explosions': gradient_explosions, 'loss_spikes': loss_spikes, 'similarity_drops': similarity_drops, 'overall': overall}
        return {'method': self.method_name, 'total_rounds': n, 'convergence': convergence, 'gradient_flow': gradient_flow, 'similarity_evolution': similarity_evolution, 'param_norms': param_norms, 'health': health}

    def save(self, prefix=''):
        if not self.enable_json:
            return
        if self.client_events:
            path = self.logs_dir / f'{prefix}client_events.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.client_events], f, indent=2)
            self.logger.info(f'Saved {len(self.client_events)} client events to {path}')
        if self.aggregation_events:
            path = self.logs_dir / f'{prefix}aggregation_events.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.aggregation_events], f, indent=2)
            self.logger.info(f'Saved {len(self.aggregation_events)} aggregation events to {path}')
        if self.debug_events:
            path = self.logs_dir / f'{prefix}debug_events.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.debug_events], f, indent=2)
            self.logger.info(f'Saved {len(self.debug_events)} debug events to {path}')
        if self.gradient_events:
            path = self.logs_dir / f'{prefix}gradient_events.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.gradient_events], f, indent=2)
            self.logger.info(f'Saved {len(self.gradient_events)} gradient events to {path}')
        if self.param_norm_events:
            path = self.logs_dir / f'{prefix}param_norms.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.param_norm_events], f, indent=2)
            self.logger.info(f'Saved {len(self.param_norm_events)} param norm events to {path}')
        if self.lr_state_events:
            path = self.logs_dir / f'{prefix}lr_states.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.lr_state_events], f, indent=2)
            self.logger.info(f'Saved {len(self.lr_state_events)} LR state events to {path}')
        if self.round_summaries:
            path = self.logs_dir / f'{prefix}round_summaries.json'
            with open(path, 'w') as f:
                json.dump([asdict(e) for e in self.round_summaries], f, indent=2)
            self.logger.info(f'Saved {len(self.round_summaries)} round summaries to {path}')
        report = self.get_experiment_report()
        if report:
            path = self.logs_dir / f'{prefix}experiment_report.json'
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
            self.logger.info(f'Saved experiment report to {path}')
        if self.enable_traces and self.round_traces:
            path = self.traces_dir / f'{prefix}round_traces.json'
            with open(path, 'w') as f:
                json.dump(self.round_traces, f, indent=2)
            self.logger.info(f'Saved traces for {len(self.round_traces)} rounds to {path}')

    def get_client_stats(self, round_num=None):
        events = self.client_events
        if round_num is not None:
            events = [e for e in events if e.round_num == round_num]
        if not events:
            return {}
        losses = [e.loss for e in events]
        return {'num_events': len(events), 'avg_loss': sum(losses) / len(losses), 'min_loss': min(losses), 'max_loss': max(losses), 'total_samples': sum((e.num_samples for e in events))}

    def get_drift_summary(self):
        drift_events = [e for e in self.debug_events if e.event_type == 'drift_metrics']
        if not drift_events:
            return {}
        m_drifts = [e.data.get('magnitude_drift', 0) for e in drift_events]
        d_drifts = [e.data.get('direction_drift', 0) for e in drift_events]
        return {'num_measurements': len(drift_events), 'avg_m_drift': sum(m_drifts) / len(m_drifts), 'avg_d_drift': sum(d_drifts) / len(d_drifts), 'max_m_drift': max(m_drifts), 'max_d_drift': max(d_drifts), 'd_to_m_ratio': sum(d_drifts) / len(d_drifts) / (sum(m_drifts) / len(m_drifts) + 1e-08)}

    def get_similarity_summary(self):
        agg_m_sims = []
        agg_a_sims = []
        agg_b_sims = []
        for e in self.aggregation_events:
            if 'm_similarity_mean' in e.custom_metrics:
                agg_m_sims.append(e.custom_metrics['m_similarity_mean'])
            if 'a_similarity_mean' in e.custom_metrics:
                agg_a_sims.append(e.custom_metrics['a_similarity_mean'])
            if 'b_similarity_mean' in e.custom_metrics:
                agg_b_sims.append(e.custom_metrics['b_similarity_mean'])
        if not agg_m_sims or not agg_a_sims or (not agg_b_sims):
            return {}
        avg_m_sim = sum(agg_m_sims) / len(agg_m_sims)
        avg_a_sim = sum(agg_a_sims) / len(agg_a_sims)
        avg_b_sim = sum(agg_b_sims) / len(agg_b_sims)
        components = [('m', avg_m_sim), ('A', avg_a_sim), ('B', avg_b_sim)]
        components.sort(key=lambda x: x[1], reverse=True)
        ordering = '>'.join((c[0] for c in components))
        fedsa_confirmed = avg_a_sim > avg_b_sim
        if avg_m_sim >= avg_a_sim:
            m_role = "m is most similar (like LoRA's A - general knowledge)"
        elif avg_m_sim >= avg_b_sim:
            m_role = 'm is between A and B'
        else:
            m_role = "m is least similar (like LoRA's B - client-specific)"
        return {'num_rounds': len(agg_m_sims), 'avg_m_similarity': avg_m_sim, 'avg_a_similarity': avg_a_sim, 'avg_b_similarity': avg_b_sim, 'ordering': ordering, 'fedsa_pattern_confirmed': fedsa_confirmed, 'm_role': m_role, 'a_minus_b': avg_a_sim - avg_b_sim, 'm_minus_a': avg_m_sim - avg_a_sim, 'm_minus_b': avg_m_sim - avg_b_sim, 'early_m_sim': sum(agg_m_sims[:5]) / min(5, len(agg_m_sims)) if agg_m_sims else 0, 'late_m_sim': sum(agg_m_sims[-5:]) / min(5, len(agg_m_sims)) if agg_m_sims else 0, 'early_a_sim': sum(agg_a_sims[:5]) / min(5, len(agg_a_sims)) if agg_a_sims else 0, 'late_a_sim': sum(agg_a_sims[-5:]) / min(5, len(agg_a_sims)) if agg_a_sims else 0, 'early_b_sim': sum(agg_b_sims[:5]) / min(5, len(agg_b_sims)) if agg_b_sims else 0, 'late_b_sim': sum(agg_b_sims[-5:]) / min(5, len(agg_b_sims)) if agg_b_sims else 0}

    def reset(self):
        self.client_events = []
        self.aggregation_events = []
        self.debug_events = []
        self.gradient_events = []
        self.param_norm_events = []
        self.lr_state_events = []
        self.round_summaries = []
        self.round_traces = {}
        self._last_val_acc = None
        self.logger.info('MethodLogger reset')
