#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
랜덤 메타데이터 기반 배치 실험 오케스트레이터.
- random_metadata.csv의 snapped_latitude/snapped_longitude를 사고 좌표로 사용
- experiment_metadata.csv에서 bbox 관련 열 제거
- snap_distance_m을 사고 좌표 옆에 기록
"""
import os
import sys
import yaml
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import subprocess

BASE_DIR = Path(__file__).resolve().parents[1]
SCE_DIR = BASE_DIR / "sce_src"
if str(SCE_DIR) not in sys.path:
    sys.path.append(str(SCE_DIR))

from make_csv_yaml_dynamic import ScenarioGenerator

def _configure_stdout():
    if sys.platform != "win32":
        return
    try:
        if sys.stdout is None or sys.stdout.closed:
            sys.stdout = sys.__stdout__
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        else:
            import io
            if hasattr(sys.stdout, "buffer") and sys.stdout.buffer:
                sys.stdout = io.TextIOWrapper(
                    sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True
                )
    except Exception:
        pass


_configure_stdout()


def _parse_indices(text):
    if not text:
        return None
    tokens = [t.strip() for t in str(text).split(",") if t.strip()]
    indices = []
    for tok in tokens:
        if "-" in tok:
            parts = [p.strip() for p in tok.split("-", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                try:
                    start = int(float(parts[0]))
                    end = int(float(parts[1]))
                except Exception:
                    continue
                if start <= end:
                    indices.extend(range(start, end + 1))
                else:
                    indices.extend(range(start, end - 1, -1))
                continue
        try:
            indices.append(int(float(tok)))
        except Exception:
            continue
    return sorted(set(indices)) if indices else None


class BatchExperimentOrchestrator:
    """랜덤 메타데이터 기반 배치 실험 오케스트레이터."""

    def __init__(self, base_path, metadata_path, config_template_path, only_indices=None):
        """
        초기화
        Args:
            base_path: 프로젝트 루트 경로
            metadata_path: random_metadata.csv 경로
            config_template_path: config.yaml 템플릿 경로
        """
        self.base_path = Path(base_path).resolve()

        # 메타데이터 로드
        try:
            metadata_path = Path(metadata_path)
            if not metadata_path.is_absolute():
                metadata_path = self.base_path / metadata_path
            self.metadata = pd.read_csv(metadata_path)
            self.metadata_path = metadata_path
            self.id_col, self.lat_col, self.lon_col, self.snap_dist_col = self._resolve_metadata_columns()
            print(f"메타데이터 로드 완료: {len(self.metadata)}건")
        except FileNotFoundError:
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {metadata_path}")
        except Exception as e:
            raise Exception(f"메타데이터 로드 실패: {e}")

        # config.yaml 로드
        try:
            with open(config_template_path, "r", encoding="utf-8") as f:
                self.config_template = yaml.safe_load(f)
            print(f"Config 템플릿 로드: {config_template_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Config 파일을 찾을 수 없습니다: {config_template_path}")
        except Exception as e:
            raise Exception(f"Config 로드 실패: {e}")

        # 실험 ID: random_metadata.csv가 들어있는 폴더명 우선 사용
        metadata_parent = Path(metadata_path).parent.name
        if metadata_parent:
            self.exp_id = metadata_parent
        else:
            self.exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 시나리오 폴더 (CSV/YAML 저장)
        self.scenarios_base = self.base_path / "scenarios" / self.exp_id
        self.scenarios_base.mkdir(parents=True, exist_ok=True)

        # 결과 폴더
        self.results_base = self.base_path / "results" / self.exp_id
        self.results_base.mkdir(parents=True, exist_ok=True)

        # 로그 폴더
        self.logs_base = self.base_path / "experiment_logs"
        self.logs_base.mkdir(parents=True, exist_ok=True)

        # 로깅 설정
        self.setup_logging()

        # 진행 상황 추적
        self.only_indices = set(only_indices or [])
        if self.only_indices:
            self.run_metadata = self.metadata[self.metadata[self.id_col].isin(self.only_indices)].copy()
        else:
            self.run_metadata = self.metadata
        self.total_points_all = len(self.metadata)
        self.total_points = len(self.run_metadata)
        self.completed_points = 0
        self.failed_points = []

        # 성능 메트릭 추적 (site_id -> {scenario_gen_time, sim_time, api_calls, status, failure_reason})
        self.performance_metrics = {}

    def setup_logging(self):
        """로깅 설정 (파일 + 콘솔)"""
        log_file = self.logs_base / f"{self.exp_id}.log"

        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def _resolve_metadata_columns(self):
        columns = set(self.metadata.columns)

        id_candidates = ["grid_id", "point_id", "index", "random_id", "id"]
        id_col = next((c for c in id_candidates if c in columns), None)
        if id_col is None:
            self.metadata = self.metadata.copy()
            self.metadata["point_id"] = range(1, len(self.metadata) + 1)
            id_col = "point_id"

        if "snapped_latitude" in columns and "snapped_longitude" in columns:
            lat_col = "snapped_latitude"
            lon_col = "snapped_longitude"
        else:
            raise ValueError(
                "random_metadata.csv에는 snapped_latitude / snapped_longitude가 있어야 합니다."
            )

        snap_dist_col = "snap_distance_m" if "snap_distance_m" in columns else None

        return id_col, lat_col, lon_col, snap_dist_col

    def _drop_bbox_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        bbox_cols = [c for c in df.columns if "bbox" in c.lower()]
        return df.drop(columns=bbox_cols, errors="ignore")

    def _reorder_coordinate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = list(df.columns)
        if self.lat_col not in cols or self.lon_col not in cols:
            return df

        coord_cols = [self.lat_col, self.lon_col]
        if self.snap_dist_col and self.snap_dist_col in cols:
            coord_cols.append(self.snap_dist_col)

        original_cols = list(df.columns)
        insert_at = min(original_cols.index(self.lat_col), original_cols.index(self.lon_col))

        for col in coord_cols:
            if col in cols:
                cols.remove(col)

        new_cols = cols[:insert_at] + coord_cols + cols[insert_at:]
        return df[new_cols]

    def save_experiment_metadata(self, include_performance=False):
        """
        실험 메타데이터 저장
        Args:
            include_performance: True이면 성능 메트릭 포함 (실험 종료 시)
        """
        metadata_path = self.scenarios_base / "experiment_metadata.csv"
        if self.only_indices and metadata_path.exists():
            metadata = pd.read_csv(metadata_path, encoding="utf-8-sig")
        else:
            metadata = self.metadata.copy()
            metadata = self._drop_bbox_columns(metadata)

            id_col = self.id_col
            metadata["experiment_id"] = self.exp_id
            metadata["incident_size"] = self.config_template["entity_info"]["patient"]["incident_size"]
            metadata["amb_velocity"] = self.config_template["entity_info"]["ambulance"]["velocity"]
            metadata["uav_velocity"] = self.config_template["entity_info"]["uav"]["velocity"]
            metadata["total_samples"] = self.config_template["run_setting"]["totalSamples"]
            metadata["random_seed"] = self.config_template["run_setting"]["random_seed"]
            metadata["is_use_time"] = self.config_template["entity_info"]["ambulance"].get("is_use_time", True)
            metadata["uav_size"] = 0

        id_col = self.id_col

        if include_performance and self.performance_metrics:
            for col in ["scenario_gen_time_sec", "simulation_time_sec", "api_call_count", "status", "failure_reason"]:
                if col not in metadata.columns:
                    metadata[col] = ""

            for gid, metrics in self.performance_metrics.items():
                mask = metadata[id_col] == gid
                if not mask.any():
                    continue
                metadata.loc[mask, "scenario_gen_time_sec"] = metrics.get("scenario_gen_time_sec", "")
                metadata.loc[mask, "simulation_time_sec"] = metrics.get("simulation_time_sec", "")
                metadata.loc[mask, "api_call_count"] = metrics.get("api_call_count", "")
                metadata.loc[mask, "status"] = metrics.get("status", "")
                metadata.loc[mask, "failure_reason"] = metrics.get("failure_reason", "")

            # Experiment-level summary columns (repeat per row)
            success_mask = metadata["status"] == "success"
            avg_scenario_gen = metadata.loc[success_mask, "scenario_gen_time_sec"].dropna().mean()
            avg_sim_time = metadata.loc[success_mask, "simulation_time_sec"].dropna().mean()
            avg_api_calls = metadata.loc[success_mask, "api_call_count"].dropna().mean()
            total_points = len(metadata)
            success_count = int(success_mask.sum())
            success_rate = (success_count / total_points * 100) if total_points > 0 else 0.0

            insert_at = metadata.columns.get_loc("failure_reason") + 1 if "failure_reason" in metadata.columns else len(metadata.columns)
            summary_cols = [
                "평균 시나리오 생성 시간(초)",
                "평균 시뮬레이션 실행 시간(초)",
                "평균 API 호출 수",
                "전체 좌표 수",
                "실험 성공 수",
                "성공률(%)",
            ]
            summary_vals = [
                avg_scenario_gen,
                avg_sim_time,
                avg_api_calls,
                total_points,
                success_count,
                success_rate,
            ]
            for col in summary_cols:
                if col not in metadata.columns:
                    metadata.insert(insert_at, col, "")
                    insert_at += 1
            if len(metadata) > 0:
                for col, val in zip(summary_cols, summary_vals):
                    metadata.at[metadata.index[0], col] = val

        metadata = self._reorder_coordinate_columns(metadata)

        metadata.to_csv(metadata_path, index=False, encoding="utf-8-sig")
        self.logger.info(f"실험 메타데이터 저장: {metadata_path}")

    def run_experiment(self):
        self.logger.info("=" * 70)
        self.logger.info(f"실험 시작: {self.exp_id}")
        self.logger.info("=" * 70)
        self.logger.info(f"총 좌표 수: {self.total_points_all}")
        self.logger.info(f"시나리오 폴더: {self.scenarios_base}")
        self.logger.info(f"결과 폴더: {self.results_base}")
        self.logger.info(f"로그 파일: {self.logs_base / f'{self.exp_id}.log'}")
        self.logger.info("=" * 70)

        # 초기 메타데이터 저장 (재시도 모드에서는 기존 파일 유지)
        metadata_path = self.scenarios_base / "experiment_metadata.csv"
        if not (self.only_indices and metadata_path.exists()):
            self.save_experiment_metadata()

        # 좌표별 처리
        for idx, row in self.run_metadata.iterrows():
            site_id = row[self.id_col]
            lat = row[self.lat_col]
            lon = row[self.lon_col]
            snap_dist = None
            if self.snap_dist_col and self.snap_dist_col in row:
                snap_dist = row[self.snap_dist_col]

            try:
                coord_label = f"({lat:.6f}, {lon:.6f})"
                if snap_dist is not None and not pd.isna(snap_dist):
                    coord_label += f", snap_distance_m={float(snap_dist):.1f}"

                self.logger.info(
                    f"[{self.completed_points + 1}/{self.total_points}] 좌표 {site_id} 처리 시작: {coord_label}"
                )
                self.process_site(site_id, lat, lon)
                self.completed_points += 1
                progress_pct = (self.completed_points / self.total_points) * 100
                self.logger.info(
                    f"좌표 {site_id} 완료 ({self.completed_points}/{self.total_points}, {progress_pct:.1f}%)"
                )
            except Exception as e:
                error_str = str(e)
                self.failed_points.append((site_id, lat, lon, error_str))

                if "NoSegment" in error_str:
                    failure_type = "NoSegment (도로 스냅 실패)"
                elif "NoRoute" in error_str:
                    failure_type = "NoRoute (경로 없음)"
                elif "timeout" in error_str.lower():
                    failure_type = "API 타임아웃"
                elif "Rate limit" in error_str or "429" in error_str:
                    failure_type = "API 호출 제한"
                else:
                    failure_type = "기타 오류"

                self.performance_metrics[site_id] = {
                    "scenario_gen_time_sec": None,
                    "simulation_time_sec": None,
                    "api_call_count": None,
                    "status": "failed",
                    "failure_reason": f"{failure_type}: {error_str[:200]}",
                }
                self.logger.error(f"좌표 {site_id} 실패 [{failure_type}]: {error_str}")
                continue

        self.print_summary()
        self.logger.info("\n[종료] 성능 메트릭 포함 메타데이터 저장 중...")
        self.save_experiment_metadata(include_performance=True)

    def process_site(self, site_id, latitude, longitude):
        """
        단일 좌표 처리
        Args:
            site_id: 좌표 ID
            latitude: 사고 위도
            longitude: 사고 경도
        """
        import time

        # 1) 시나리오 생성 (CSV + YAML)
        self.logger.info("  시나리오 생성 중...")
        scenario_gen_start = time.time()

        generator = ScenarioGenerator(
            base_path=str(self.base_path),
            experiment_id=self.exp_id,
        )

        incident_size = self.config_template["entity_info"]["patient"]["incident_size"]
        amb_velocity = self.config_template["entity_info"]["ambulance"]["velocity"]
        uav_velocity = self.config_template["entity_info"]["uav"]["velocity"]
        total_samples = self.config_template["run_setting"]["totalSamples"]
        random_seed = self.config_template["run_setting"]["random_seed"]
        is_use_time = self.config_template["entity_info"]["ambulance"].get("is_use_time", True)
        duration_coeff = self.config_template["entity_info"]["ambulance"].get("duration_coeff", 1.0)

        config_path, api_calls = generator.generate_scenario(
            latitude=latitude,
            longitude=longitude,
            incident_size=incident_size,
            amb_size=incident_size,
            uav_size=0,
            amb_velocity=amb_velocity,
            uav_velocity=uav_velocity,
            total_samples=total_samples,
            random_seed=random_seed,
            is_use_time=is_use_time,
            amb_handover_time=0,
            uav_handover_time=0,
            duration_coeff=duration_coeff,
        )

        scenario_gen_time = round(time.time() - scenario_gen_start, 2)
        self.logger.info(
            f"  시나리오 생성 완료 ({scenario_gen_time}초, API 호출: {api_calls}회)"
        )

        # 2) 시뮬레이션 실행
        self.logger.info("  시뮬레이션 실행 중...")
        sim_start = time.time()
        result = subprocess.run(
            [
                sys.executable,
                str(self.base_path / "sim_src" / "main.py"),
                "--config_path",
                config_path,
            ],
            cwd=str(self.base_path.parent),
            capture_output=True,
            text=True,
            timeout=1200,  # 20분 제한
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            raise RuntimeError(f"시뮬레이션 실패: {result.stderr}")

        sim_time = round(time.time() - sim_start, 2)
        self.logger.info(f"  시뮬레이션 완료 ({sim_time}초)")

        # 3) 성능 메트릭 저장
        self.performance_metrics[site_id] = {
            "scenario_gen_time_sec": scenario_gen_time,
            "simulation_time_sec": sim_time,
            "api_call_count": api_calls,
            "status": "success",
            "failure_reason": None,
        }

    def print_summary(self):
        self.logger.info("=" * 70)
        self.logger.info(f"실험 요약: {self.exp_id}")
        self.logger.info("=" * 70)
        self.logger.info(f"성공: {self.completed_points}/{self.total_points}")
        self.logger.info(f"실패: {len(self.failed_points)}")

        if self.failed_points:
            failure_types = {}
            for site_id, lat, lon, error in self.failed_points:
                if "NoSegment" in error:
                    failure_type = "NoSegment (도로 스냅 실패)"
                elif "NoRoute" in error:
                    failure_type = "NoRoute (경로 없음)"
                elif "timeout" in error.lower():
                    failure_type = "API 타임아웃"
                elif "Rate limit" in error or "429" in error:
                    failure_type = "API 호출 제한"
                else:
                    failure_type = "기타 오류"
                failure_types.setdefault(failure_type, []).append((site_id, lat, lon, error))

            self.logger.info("실패 유형:")
            for failure_type, items in failure_types.items():
                self.logger.info(f"  - {failure_type}: {len(items)}건")

            self.logger.info("실패 좌표 예시 (최대 10건):")
            for site_id, lat, lon, error in self.failed_points[:10]:
                short_error = error[:100] + "..." if len(error) > 100 else error
                self.logger.info(f"  - {site_id} ({lat:.6f}, {lon:.6f}): {short_error}")

            if len(self.failed_points) > 10:
                self.logger.info(f"  ... {len(self.failed_points) - 10}건 추가")

        success_rate = (self.completed_points / self.total_points * 100) if self.total_points > 0 else 0
        self.logger.info(f"성공률: {success_rate:.1f}%")
        self.logger.info("=" * 70)


def main():
    """메인 진입점"""
    import argparse

    parser = argparse.ArgumentParser(description="랜덤 메타데이터 기반 배치 실험")
    parser.add_argument("--base_path", required=True, help="프로젝트 루트 경로")
    parser.add_argument(
        "--random_metadata",
        dest="metadata_path",
        required=True,
        help="random_metadata.csv 경로",
    )
    parser.add_argument("--config_template", default="config.yaml", help="config 템플릿 경로")
    parser.add_argument(
        "--only_indices",
        default=None,
        help="재시도할 인덱스 목록 (예: 23,97,160 또는 1-5,10)",
    )
    args = parser.parse_args()

    try:
        orchestrator = BatchExperimentOrchestrator(
            base_path=args.base_path,
            metadata_path=args.metadata_path,
            config_template_path=args.config_template,
            only_indices=_parse_indices(args.only_indices),
        )
        orchestrator.run_experiment()
    except Exception as e:
        print(f"\n실행 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
