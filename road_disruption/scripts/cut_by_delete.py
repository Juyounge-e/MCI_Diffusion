"""
도로를 pbf 에서 아예 지우고 별도 OSRM 인스턴스로 띄운다.

    python road_disruption/scripts/cut_by_delete.py \
        --spec config/cuts/daejeon_major_bridges.yaml --port 5010

흐름:
    지역 pbf ──(대상 way 삭제)──> 차단본 pbf
                                     │  osrm-extract/partition/customize
                                     ▼
                              차단본 OSRM (별 포트)   ← baseline(5002)은 그대로

baseline 과 차단본이 동시에 살아 있으므로 reset 이 필요 없다.
같은 OD 를 두 포트에 물어보면 그게 곧 차단 전/후 비교다.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roaddis import osrm
from roaddis.cli import slug
from roaddis.cutspec import CutSpec
from roaddis.paths import DATA_DIR, OSRM_DIR, PKG_DIR
from roaddis.pbf import write_pbf_without_ways
from roaddis.regions import get_region
from roaddis.way_index import WayIndex


def main():
    ap = argparse.ArgumentParser(description="way 삭제 방식 도로 차단")
    ap.add_argument("--spec", required=True)
    ap.add_argument("--port", type=int, default=5010)
    ap.add_argument("--way-mode", default="any", choices=["any", "all"],
                    help="any: 세그먼트가 하나라도 선택되면 way 전체 삭제(기본). "
                         "all: way 의 모든 세그먼트가 선택된 경우만 삭제.")
    ap.add_argument("--keep-pbf", action="store_true", help="차단본 pbf 를 남긴다")
    args = ap.parse_args()

    sp = Path(args.spec)
    if not sp.is_absolute() and not sp.exists():
        sp = PKG_DIR / args.spec
    spec = CutSpec.from_yaml(sp)
    region = get_region(spec.region)

    t0 = time.time()
    print(f"[1/4] CutSpec '{spec.name}' 해석  (way-mode={args.way_mode})")
    idx = WayIndex.load(DATA_DIR / "way_index" / spec.region)
    seg_mask = spec.resolve(idx)
    way_pos = idx.segment_way_pos()

    n_sel = np.bincount(way_pos[seg_mask], minlength=len(idx))
    n_tot = np.bincount(way_pos, minlength=len(idx))
    if args.way_mode == "any":
        drop_mask = n_sel > 0
    else:
        drop_mask = (n_sel > 0) & (n_sel == n_tot)

    drop_ids = set(int(w) for w in idx.way_id[drop_mask])
    partial = int(((n_sel > 0) & (n_sel < n_tot)).sum())
    print(f"      선택 세그먼트 {int(seg_mask.sum()):,} → 삭제 way {len(drop_ids):,}개")
    if partial:
        print(f"      [주의] 일부 세그먼트만 선택된 way {partial:,}개 — "
              f"{'전체를 삭제합니다' if args.way_mode=='any' else '삭제하지 않습니다'}")

    if not drop_ids:
        sys.exit("삭제할 way 가 없습니다.")

    # CutSpec 이름이 이미 지역명으로 시작하면 중복시키지 않는다
    nm = slug(spec.name)
    name = nm if nm.startswith(spec.region) else f"{spec.region}_{nm}"
    cut_pbf_rel = f"cut_pbf/{name}.osm.pbf"
    cut_pbf = OSRM_DIR / cut_pbf_rel

    print(f"[2/4] pbf 재작성 → {cut_pbf.name}")
    st = write_pbf_without_ways(OSRM_DIR / region.pbf, cut_pbf, drop_ids)
    print(f"      node {st['nodes']:,}  way {st['ways_kept']:,} 유지 / "
          f"{st['ways_dropped']:,} 삭제  relation {st['relations']:,}")
    print(f"      {cut_pbf.stat().st_size/1e6:.1f} MB")

    print(f"[3/4] OSRM 빌드 (extract → partition → customize)")
    osrm.extract(cut_pbf_rel)
    osrm.partition(cut_pbf_rel)
    osrm.customize(cut_pbf_rel)

    container = f"osrm-cut-{slug(spec.name)}"[:60]
    print(f"[4/4] 기동  container={container}  port={args.port}")
    osrm.serve(cut_pbf_rel, args.port, container)
    url = f"http://localhost:{args.port}"
    if not osrm.wait_ready(url):
        sys.exit(f"서버 응답 없음. docker logs {container}")

    print(f"완료 ({time.time()-t0:.1f}s)")
    print()
    print(f"  baseline : {region.osrm_url}")
    print(f"  차단본   : {url}")
    print(f"  두 포트에 같은 OD 를 물어보면 그게 차단 전/후 비교입니다.")
    print(f"  정리: docker rm -f {container}"
          + ("" if args.keep_pbf else f"   (pbf: {cut_pbf})"))


if __name__ == "__main__":
    main()
