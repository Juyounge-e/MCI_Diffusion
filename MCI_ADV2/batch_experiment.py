#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
배치 실험 오케스트레이터

대전 그리드 전체에 대해 시나리오 생성 및 시뮬레이션 실행
"""

import os
import sys
import yaml
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from make_csv_yaml_dynamic import ScenarioGenerator
import subprocess

# Windows 콘솔 UTF-8 설정
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


class BatchExperimentOrchestrator:
    """배치 실험 오케스트레이터"""

    def __init__(self, base_path, grid_metadata_path, config_template_path):
        """
        초기화

        Args:
            base_path: 프로젝트 루트 경로
            grid_metadata_path: grid_metadata.csv 경로
            config_template_path: config.yaml 템플릿 경로
        """
        self.base_path = Path(base_path)

        # Grid metadata 로드
        try:
            self.grid_metadata = pd.read_csv(grid_metadata_path)
            print(f"✅ Grid metadata 로드: {len(self.grid_metadata)}개 그리드")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Grid metadata 파일을 찾을 수 없습니다: {grid_metadata_path}")
        except Exception as e:
            raise Exception(f"❌ Grid metadata 로드 실패: {e}")

        # config.yaml 로드
        try:
            with open(config_template_path, 'r', encoding='utf-8') as f:
                self.config_template = yaml.safe_load(f)
            print(f"✅ Config 템플릿 로드: {config_template_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ Config 파일을 찾을 수 없습니다: {config_template_path}")
        except Exception as e:
            raise Exception(f"❌ Config 로드 실패: {e}")

        # 실험 ID 생성: exp_YYYYMMDD_HHMMSS 형식 (통일)
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
        self.total_grids = len(self.grid_metadata)
        self.completed_grids = 0
        self.failed_grids = []

        # 성능 메트릭 추적 (grid_id -> {scenario_gen_time, sim_time, api_calls, status, failure_reason})
        self.performance_metrics = {}

    def setup_logging(self):
        """로깅 설정 (파일 + 콘솔)"""
        log_file = self.logs_base / f"{self.exp_id}.log"

        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def save_experiment_metadata(self, include_performance=False):
        """실험 메타데이터 저장

        Args:
            include_performance: True이면 성능 메트릭 포함 (실험 종료 후)
        """
        # 그리드 메타데이터에 파라미터 추가
        metadata = self.grid_metadata.copy()
        metadata['experiment_id'] = self.exp_id
        metadata['incident_size'] = self.config_template['entity_info']['patient']['incident_size']
        metadata['amb_velocity'] = self.config_template['entity_info']['ambulance']['velocity']
        metadata['uav_velocity'] = self.config_template['entity_info']['uav']['velocity']
        metadata['total_samples'] = self.config_template['run_setting']['totalSamples']
        metadata['random_seed'] = self.config_template['run_setting']['random_seed']
        metadata['is_use_time'] = self.config_template['entity_info']['ambulance'].get('is_use_time', True)
        metadata['uav_size'] = 0  # 고정값

        # 성능 메트릭 추가 (실험 종료 후)
        if include_performance and self.performance_metrics:
            metadata['scenario_gen_time_sec'] = metadata['grid_id'].map(
                lambda gid: self.performance_metrics.get(gid, {}).get('scenario_gen_time_sec', None)
            )
            metadata['simulation_time_sec'] = metadata['grid_id'].map(
                lambda gid: self.performance_metrics.get(gid, {}).get('simulation_time_sec', None)
            )
            metadata['api_call_count'] = metadata['grid_id'].map(
                lambda gid: self.performance_metrics.get(gid, {}).get('api_call_count', None)
            )
            metadata['status'] = metadata['grid_id'].map(
                lambda gid: self.performance_metrics.get(gid, {}).get('status', 'not_processed')
            )
            metadata['failure_reason'] = metadata['grid_id'].map(
                lambda gid: self.performance_metrics.get(gid, {}).get('failure_reason', None)
            )

            # Experiment-level summary columns (repeat per row)
            success_mask = metadata['status'] == 'success'
            avg_scenario_gen = metadata.loc[success_mask, 'scenario_gen_time_sec'].dropna().mean()
            avg_sim_time = metadata.loc[success_mask, 'simulation_time_sec'].dropna().mean()
            avg_api_calls = metadata.loc[success_mask, 'api_call_count'].dropna().mean()
            total_grids = len(metadata)
            success_count = int(success_mask.sum())
            success_rate = (success_count / total_grids * 100) if total_grids > 0 else 0.0

            insert_at = metadata.columns.get_loc('failure_reason') + 1
            summary_cols = [
                '평균 시나리오 생성 시간',
                '평균 시뮬레이션 실행 시간',
                '평균 API 호출 건 수',
                '전체 그리드 수',
                '실험 성공 수',
                '성공율'
            ]
            summary_vals = [
                avg_scenario_gen,
                avg_sim_time,
                avg_api_calls,
                total_grids,
                success_count,
                success_rate
            ]
            for offset, (col, val) in enumerate(zip(summary_cols, summary_vals)):
                metadata.insert(insert_at + offset, col, val)

        # 저장
        metadata_path = self.scenarios_base / "experiment_metadata.csv"
        metadata.to_csv(metadata_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"📋 실험 메타데이터 저장: {metadata_path}")

    def run_experiment(self):
        """메인 실험 루프"""
        self.logger.info(f"{'='*70}")
        self.logger.info(f"🚀 배치 실험 시작: {self.exp_id}")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"총 그리드 수: {self.total_grids}")
        self.logger.info(f"시나리오 저장: {self.scenarios_base}")
        self.logger.info(f"결과 저장: {self.results_base}")
        self.logger.info(f"로그 저장: {self.logs_base / f'{self.exp_id}.log'}")
        self.logger.info(f"{'='*70}\n")

        # 메타데이터 저장
        self.save_experiment_metadata()

        # 각 그리드 처리
        for idx, row in self.grid_metadata.iterrows():
            grid_id = row['grid_id']
            lat = row['latitude']
            lon = row['longitude']

            try:
                self.logger.info(f"\n[{idx+1}/{self.total_grids}] 🔄 그리드 {grid_id} 처리 중: ({lat:.6f}, {lon:.6f})")

                self.process_grid(grid_id, lat, lon)

                self.completed_grids += 1
                progress_pct = (self.completed_grids / self.total_grids) * 100
                self.logger.info(f"✅ 그리드 {grid_id} 완료 ({self.completed_grids}/{self.total_grids}, {progress_pct:.1f}%)")

            except Exception as e:
                error_str = str(e)
                self.failed_grids.append((grid_id, lat, lon, error_str))

                # 실패 원인 분류
                if "NoSegment" in error_str or "내에 도로 없음" in error_str:
                    failure_type = "격오지 (도로 없음)"
                elif "NoRoute" in error_str or "도로 연결 안됨" in error_str:
                    failure_type = "도로 연결 안됨"
                elif "API 타임아웃" in error_str or "timeout" in error_str.lower():
                    failure_type = "API 타임아웃"
                elif "Rate limit" in error_str:
                    failure_type = "API Rate Limit 초과"
                else:
                    failure_type = "기타 오류"

                # 실패 메트릭 기록
                self.performance_metrics[grid_id] = {
                    'scenario_gen_time_sec': None,
                    'simulation_time_sec': None,
                    'api_call_count': None,
                    'status': 'failed',
                    'failure_reason': f"{failure_type}: {error_str[:200]}"  # 최대 200자
                }

                self.logger.error(f"❌ 그리드 {grid_id} 실패 [{failure_type}]: {error_str}")
                continue

        # 최종 요약
        self.print_summary()

        # 성능 메트릭을 포함한 메타데이터 재저장
        self.logger.info(f"\n📊 성능 메트릭을 포함한 메타데이터 저장 중...")
        self.save_experiment_metadata(include_performance=True)

    def process_grid(self, grid_id, latitude, longitude):
        """
        단일 그리드 처리

        Args:
            grid_id: 그리드 ID
            latitude: 그리드 중심점 위도
            longitude: 그리드 중심점 경도
        """
        import time

        # 1. 시나리오 생성 (CSV + YAML)
        self.logger.info(f"  📁 시나리오 생성 중...")
        scenario_gen_start = time.time()

        generator = ScenarioGenerator(
            base_path=str(self.base_path),
            experiment_id=self.exp_id
        )

        # config.yaml에서 파라미터 추출
        incident_size = self.config_template['entity_info']['patient']['incident_size']
        amb_velocity = self.config_template['entity_info']['ambulance']['velocity']
        uav_velocity = self.config_template['entity_info']['uav']['velocity']
        total_samples = self.config_template['run_setting']['totalSamples']
        random_seed = self.config_template['run_setting']['random_seed']
        is_use_time = self.config_template['entity_info']['ambulance'].get('is_use_time', True)
        duration_coeff = self.config_template['entity_info']['ambulance'].get('duration_coeff', 1.0)

        # 시나리오 생성 (routes 폴더 없음)
        config_path, api_calls = generator.generate_scenario(
            latitude=latitude,
            longitude=longitude,
            incident_size=incident_size,
            amb_size=incident_size,  # 기존 파라미터 사용
            uav_size=0,  # UAV 0대 고정
            amb_velocity=amb_velocity,
            uav_velocity=uav_velocity,
            total_samples=total_samples,
            random_seed=random_seed,
            is_use_time=is_use_time,
            amb_handover_time=0,
            uav_handover_time=0,
            duration_coeff=duration_coeff
        )

        scenario_gen_time = round(time.time() - scenario_gen_start, 2)
        self.logger.info(f"  ✅ 시나리오 생성 완료 ({scenario_gen_time}초, API 호출: {api_calls}회)")

        # 2. 시뮬레이션 실행
        self.logger.info(f"  🎮 시뮬레이션 실행 중...")
        sim_start = time.time()

        result = subprocess.run(
            [sys.executable, str(self.base_path / "main.py"),
             "--config_path", config_path],
            cwd=str(self.base_path),
            capture_output=True,
            text=True,
            timeout=1200,  # 20분 타임아웃
            encoding='utf-8'
        )

        if result.returncode != 0:
            raise RuntimeError(f"시뮬레이션 실패: {result.stderr}")

        sim_time = round(time.time() - sim_start, 2)
        self.logger.info(f"  ✅ 시뮬레이션 완료 ({sim_time}초)")

        # 3. 성능 메트릭 저장
        self.performance_metrics[grid_id] = {
            'scenario_gen_time_sec': scenario_gen_time,
            'simulation_time_sec': sim_time,
            'api_call_count': api_calls,
            'status': 'success',
            'failure_reason': None
        }

    def print_summary(self):
        """최종 요약 출력"""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"📊 실험 {self.exp_id} 완료")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"✅ 성공: {self.completed_grids}/{self.total_grids}")
        self.logger.info(f"❌ 실패: {len(self.failed_grids)}")

        if self.failed_grids:
            # 실패 유형별 분류
            failure_types = {}
            for grid_id, lat, lon, error in self.failed_grids:
                if "NoSegment" in error or "내에 도로 없음" in error:
                    failure_type = "격오지 (도로 없음)"
                elif "NoRoute" in error or "도로 연결 안됨" in error:
                    failure_type = "도로 연결 안됨"
                elif "API 타임아웃" in error or "timeout" in error.lower():
                    failure_type = "API 타임아웃"
                elif "Rate limit" in error:
                    failure_type = "API Rate Limit 초과"
                else:
                    failure_type = "기타 오류"

                if failure_type not in failure_types:
                    failure_types[failure_type] = []
                failure_types[failure_type].append((grid_id, lat, lon, error))

            self.logger.info(f"\n실패 유형별 통계:")
            for failure_type, grids in failure_types.items():
                self.logger.info(f"  • {failure_type}: {len(grids)}건")

            self.logger.info(f"\n실패한 그리드 상세:")
            for grid_id, lat, lon, error in self.failed_grids[:10]:  # 최대 10개만 출력
                short_error = error[:100] + "..." if len(error) > 100 else error
                self.logger.info(f"  - 그리드 {grid_id} ({lat:.6f}, {lon:.6f}): {short_error}")
            if len(self.failed_grids) > 10:
                self.logger.info(f"  ... 외 {len(self.failed_grids) - 10}개 (전체 목록은 메타데이터 CSV 참조)")

        success_rate = (self.completed_grids / self.total_grids * 100) if self.total_grids > 0 else 0
        self.logger.info(f"\n성공률: {success_rate:.1f}%")
        self.logger.info(f"{'='*70}\n")


def main():
    """메인 진입점"""
    import argparse
    parser = argparse.ArgumentParser(description="대전 그리드 배치 실험")
    parser.add_argument("--base_path", required=True, help="프로젝트 루트 경로")
    parser.add_argument("--grid_metadata", required=True, help="grid_metadata.csv 경로")
    parser.add_argument("--config_template", default="config.yaml", help="config 템플릿 경로")
    args = parser.parse_args()

    try:
        orchestrator = BatchExperimentOrchestrator(
            base_path=args.base_path,
            grid_metadata_path=args.grid_metadata,
            config_template_path=args.config_template
        )

        orchestrator.run_experiment()

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
