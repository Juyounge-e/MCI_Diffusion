import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from make_csv_yaml_dynamic import ScenarioGenerator

TRAFFIC_COLORS = {
    0: "#7f8c8d",  # 정보 없음
    1: "#2ecc71",  # 원활
    2: "#f1c40f",  # 서행
    3: "#e74c3c",  # 혼잡
    4: "#c0392b",  # 매우 혼잡
}

KST = timezone(timedelta(hours=9))


def list_experiments(base_path: Path):
    scenarios = base_path / "scenarios"
    if not scenarios.exists():
        return []
    return sorted([p.name for p in scenarios.iterdir() if p.is_dir() and p.name.startswith("exp_")])


def list_coord_folders(exp_path: Path):
    if not exp_path.exists():
        return []
    return sorted([p.name for p in exp_path.iterdir() if p.is_dir()])


def resolve_coord_base(base_path: Path, exp_name: str, mode: str):
    if not exp_name:
        return Path()
    exp_base = base_path / "scenarios" / exp_name
    mode_dir = exp_base / mode if mode else None
    if mode_dir and mode_dir.exists():
        return mode_dir
    return exp_base


def resolve_routes_dir(exp_base: Path, coord_name: str, route_type: str, provider: str):
    if not coord_name:
        return Path()
    base_dir = exp_base / coord_name / "routes" / route_type
    if provider:
        provider_dir = base_dir / provider
        if provider_dir.exists():
            return provider_dir
    if base_dir.exists():
        return base_dir
    return base_dir


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()


def detect_provider(payload):
    if not payload:
        return "unknown"
    if "kakao_response" in payload:
        return "kakao"
    if "osrm_response" in payload:
        return "osrm"
    return "unknown"


def parse_source_index(meta, filename):
    if meta and meta.get("source_index") is not None:
        return meta.get("source_index")
    m = re.match(r"^(\\d+)_", filename)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def load_routes(routes_dir: Path):
    provider_map = {}
    if not routes_dir.exists():
        return provider_map
    for path in routes_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = data.get("meta", {})
        payload = data.get("payload", {})
        provider = meta.get("api_provider") or detect_provider(payload)
        source_index = parse_source_index(meta, path.name)
        name = meta.get("name") or path.stem
        provider_map.setdefault(provider, {})[source_index] = {
            "path": path,
            "meta": meta,
            "payload": payload,
            "name": name,
        }
    return provider_map


def load_hospital_type_map(exp_base: Path, coord_name: str):
    if not coord_name:
        return {}
    csv_path = exp_base / coord_name / "hospital_info_road.csv"
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")
    if df.empty:
        return {}
    df.columns = [str(col).strip() for col in df.columns]
    if "요양기관명" not in df.columns or "종별코드" not in df.columns:
        return {}
    type_map = {}
    for _, row in df.iterrows():
        name = str(row.get("요양기관명", "")).strip()
        if not name:
            continue
        try:
            type_map[name] = int(row.get("종별코드"))
        except (TypeError, ValueError):
            continue
    return type_map


def extract_endpoints(meta):
    if not meta:
        return None, None
    note = meta.get("direction_note", "")
    if "->" in note:
        start_key, end_key = note.split("->", 1)
        start = meta.get(start_key)
        end = meta.get(end_key)
        return start, end
    coord_keys = []
    for key, value in meta.items():
        if isinstance(value, list) and len(value) == 2:
            coord_keys.append((key, value))
    if len(coord_keys) >= 2:
        return coord_keys[0][1], coord_keys[1][1]
    return None, None


def format_route_label(routes, idx):
    name = routes.get(idx, {}).get("name")
    return f"{idx:03d} - {name}" if name else f"{idx:03d}"


def marker_html(symbol, color=None):
    if color:
        return f"<div style='font-size:24px; line-height:24px; color:{color}; font-weight:700;'>{symbol}</div>"
    return f"<div style='font-size:24px; line-height:24px;'>{symbol}</div>"


def add_div_marker(m, location, html, tooltip=None):
    folium.Marker(
        location=location,
        icon=folium.DivIcon(html=html, icon_size=(24, 24), icon_anchor=(12, 12)),
        tooltip=tooltip,
    ).add_to(m)


def hospital_marker_color(hospital_name, hospital_type_map):
    if not hospital_name or not hospital_type_map:
        return "#f1c40f"
    code = hospital_type_map.get(hospital_name)
    if code == 1:
        return "#e74c3c"
    if code == 11:
        return "#f39c12"
    return "#f1c40f"


def add_route_markers(m, route_record, hospital_type_map=None):
    meta = route_record.get("meta", {})
    route_type = meta.get("route_type")
    start, end = extract_endpoints(meta)
    if not start or not end:
        return
    start_loc = [start[1], start[0]]
    end_loc = [end[1], end[0]]
    hospital_type_map = hospital_type_map or {}

    if route_type == "center2site":
        add_div_marker(m, start_loc, marker_html("🚑"), tooltip="구급차 출발")
        add_div_marker(m, end_loc, marker_html("🔥"), tooltip="사고 지점")
    elif route_type == "hos2site":
        add_div_marker(m, start_loc, marker_html("🔥"), tooltip="사고 지점")
        hospital_name = (meta.get("name") or "").strip()
        color = hospital_marker_color(hospital_name, hospital_type_map)
        label = hospital_name or "병원"
        add_div_marker(m, end_loc, marker_html("✚", color), tooltip=label)
    else:
        folium.Marker(location=start_loc, popup="출발").add_to(m)
        folium.Marker(location=end_loc, popup="도착").add_to(m)


def make_kakao_map(route_record, hospital_type_map=None):
    payload = route_record["payload"].get("kakao_response", {})
    routes = payload.get("routes", [])
    if not routes:
        return None, "카카오 경로 데이터가 없습니다."
    route = routes[0]
    sections = route.get("sections", [])
    if not sections:
        return None, "카카오 구간 데이터가 없습니다."
    roads = sections[0].get("roads", [])
    if not roads:
        return None, "카카오 도로 데이터가 없습니다."

    points = []
    for road in roads:
        vertexes = road.get("vertexes", [])
        road_points = []
        for i in range(0, len(vertexes), 2):
            lon = vertexes[i]
            lat = vertexes[i + 1]
            road_points.append((lat, lon))
            points.append((lat, lon))

    if not points:
        return None, "카카오 경로 좌표가 없습니다."

    center = points[len(points) // 2]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    for road in roads:
        vertexes = road.get("vertexes", [])
        road_points = []
        for i in range(0, len(vertexes), 2):
            road_points.append((vertexes[i + 1], vertexes[i]))
        color = TRAFFIC_COLORS.get(road.get("traffic_state", 0), "#7f8c8d")
        if len(road_points) >= 2:
            folium.PolyLine(road_points, color=color, weight=5, opacity=0.9).add_to(m)

    add_route_markers(m, route_record, hospital_type_map=hospital_type_map)

    m.fit_bounds(points)
    return m, None


def make_osrm_map(route_record, hospital_type_map=None):
    payload = route_record["payload"].get("osrm_response", {})
    routes = payload.get("routes", [])
    if not routes:
        return None, "OSRM 경로 데이터가 없습니다."
    geometry = routes[0].get("geometry", {})
    coords = geometry.get("coordinates")
    if not coords:
        return None, "OSRM 경로 좌표가 없습니다. save_routes_json 옵션으로 다시 생성하세요."

    points = [(lat, lon) for lon, lat in coords]
    center = points[len(points) // 2]
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
    folium.PolyLine(points, color="#2980b9", weight=5, opacity=0.9).add_to(m)

    add_route_markers(m, route_record, hospital_type_map=hospital_type_map)

    m.fit_bounds(points)
    return m, None


def render_generate():
    st.header("생성")
    base_path = st.text_input("기본 경로", value=str(Path(__file__).resolve().parent), key="gen_base_path")

    if "kakao_api_key" not in st.session_state:
        st.session_state.kakao_api_key = ""

    key_input = st.text_input("카카오 API 키(선택)", value=st.session_state.kakao_api_key, type="password")
    if st.button("API 키 확인/저장"):
        st.session_state.kakao_api_key = key_input.strip()
        st.success("카카오 API 키가 저장되었습니다.")

    st.markdown("#### 📍 출발 시각 설정")
    if "departure_date_value" not in st.session_state:
        st.session_state.departure_date_value = datetime.now(KST).date()
    if "departure_time_value" not in st.session_state:
        st.session_state.departure_time_value = datetime.now(KST).time()

    col_date, col_time = st.columns(2)
    with col_date:
        departure_date = st.date_input(
            "출발 날짜",
            value=st.session_state.departure_date_value,
            help="사고 발생 예상 날짜",
            key="departure_date_input",
        )
    with col_time:
        departure_time = st.time_input(
            "출발 시각",
            value=st.session_state.departure_time_value,
            help="사고 발생 예상 시각",
            key="departure_time_input",
        )

    if departure_date != st.session_state.departure_date_value:
        st.session_state.departure_date_value = departure_date
    if departure_time != st.session_state.departure_time_value:
        st.session_state.departure_time_value = departure_time

    departure_time_str = (
        f"{st.session_state.departure_date_value.strftime('%Y%m%d')}"
        f"{st.session_state.departure_time_value.strftime('%H%M')}"
    )
    st.caption(f"→ API 파라미터: `{departure_time_str}`")

    with st.form("generate_form"):
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("위도", value=36.35, format="%.6f")
            incident_size = st.number_input("환자 수", value=30, min_value=1, step=1)
            amb_size = st.number_input("구급차 수", value=30, min_value=1, step=1)
            amb_velocity = st.number_input("구급차 속도", value=40, min_value=1, step=1)
            total_samples = st.number_input("반복 횟수", value=10, min_value=1, step=1)
        with col2:
            longitude = st.number_input("경도", value=127.38, format="%.6f")
            uav_size = st.number_input("UAV 수", value=0, min_value=0, step=1)
            uav_velocity = st.number_input("UAV 속도", value=80, min_value=1, step=1)
            random_seed = st.number_input("랜덤 시드", value=0, min_value=0, step=1)
            experiment_id = st.text_input("실험 ID(선택)", value="")
        mode_label = st.selectbox("경로 생성 모드", ["OSRM", "카카오", "둘 다"])
        if mode_label in ("카카오", "둘 다") and not st.session_state.kakao_api_key and not key_input.strip():
            st.info("카카오 모드는 API 키가 필요합니다. 키가 없으면 직선거리로 대체됩니다.")
        save_routes = st.checkbox("경로 JSON 저장", value=True)
        submit = st.form_submit_button("시나리오 생성")

    if submit:
        base_path_obj = Path(base_path)
        if not base_path_obj.exists():
            st.error("기본 경로가 존재하지 않습니다.")
            return
        kakao_key = st.session_state.kakao_api_key or key_input.strip()

        with st.spinner("시나리오 생성 중..."):
            generator = ScenarioGenerator(
                base_path,
                experiment_id or None,
                kakao_api_key=kakao_key or None,
                departure_time=departure_time_str,
            )
            coord_folder = f"lat{latitude:.6f}_lon{longitude:.6f}"
            mode_map = {"OSRM": "osrm", "카카오": "kakao", "둘 다": "both"}
            route_mode = mode_map.get(mode_label, "osrm")

            if route_mode == "both":
                config_paths = {}
                api_calls_total = 0
                for mode in ("osrm", "kakao"):
                    config_path, api_calls = generator.generate_scenario(
                        latitude=latitude,
                        longitude=longitude,
                        incident_size=int(incident_size),
                        amb_size=int(amb_size),
                        uav_size=int(uav_size),
                        amb_velocity=int(amb_velocity),
                        uav_velocity=int(uav_velocity),
                        total_samples=int(total_samples),
                        random_seed=int(random_seed),
                        is_use_time=True,
                        amb_handover_time=0,
                        uav_handover_time=0,
                        duration_coeff=1.0,
                        save_routes_json=save_routes,
                        route_mode=mode,
                        use_route_mode_subdir=True,
                    )
                    config_paths[mode] = config_path
                    api_calls_total += api_calls

                st.success("시나리오 생성 완료.")
                st.write(f"Config(OSRM): {config_paths.get('osrm')}")
                st.write(f"Config(카카오): {config_paths.get('kakao')}")
                st.write(f"API 호출 횟수(합계): {api_calls_total}")

                st.session_state.last_osrm_base = base_path
                st.session_state.last_osrm_exp = generator.experiment_id
                st.session_state.last_osrm_coord = coord_folder
                st.session_state.last_kakao_base = base_path
                st.session_state.last_kakao_exp = generator.experiment_id
                st.session_state.last_kakao_coord = coord_folder
            else:
                config_path, api_calls = generator.generate_scenario(
                    latitude=latitude,
                    longitude=longitude,
                    incident_size=int(incident_size),
                    amb_size=int(amb_size),
                    uav_size=int(uav_size),
                    amb_velocity=int(amb_velocity),
                    uav_velocity=int(uav_velocity),
                    total_samples=int(total_samples),
                    random_seed=int(random_seed),
                    is_use_time=True,
                    amb_handover_time=0,
                    uav_handover_time=0,
                    duration_coeff=1.0,
                    save_routes_json=save_routes,
                    route_mode=route_mode,
                    use_route_mode_subdir=True,
                )
                st.success("시나리오 생성 완료.")
                st.write(f"Config 경로: {config_path}")
                st.write(f"API 호출 횟수: {api_calls}")

                if route_mode == "kakao":
                    st.session_state.last_kakao_base = base_path
                    st.session_state.last_kakao_exp = generator.experiment_id
                    st.session_state.last_kakao_coord = coord_folder
                else:
                    st.session_state.last_osrm_base = base_path
                    st.session_state.last_osrm_exp = generator.experiment_id
                    st.session_state.last_osrm_coord = coord_folder

            st.session_state["next_page"] = "시각화"
            safe_rerun()


def render_visualize():
    st.header("시각화")

    default_root = Path(__file__).resolve().parent
    viz_base = st.text_input("시각화 기본 경로", value=str(default_root), key="viz_base_path")
    sync_base = st.checkbox("카카오/OSRM 경로를 기본 경로와 동기화", value=True, key="viz_sync_base")
    base_path_value = viz_base.strip() or str(default_root)

    if sync_base:
        st.session_state.kakao_base_path = base_path_value
        st.session_state.osrm_base_path = base_path_value

    default_osrm_base = Path(base_path_value)
    default_kakao_base = Path("C:/Users/User/MCI_ADV/Simul_team")
    if sync_base:
        default_kakao_base = default_osrm_base
    elif not default_kakao_base.exists():
        default_kakao_base = default_osrm_base

    route_type = st.sidebar.selectbox("경로 유형", ["center2site", "hos2site"], key="route_type")

    with st.sidebar.expander("카카오 소스", expanded=True):
        kakao_default = st.session_state.get("last_kakao_base", str(default_kakao_base))
        if sync_base:
            kakao_default = str(default_kakao_base)
        kakao_base = st.text_input(
            "카카오 기본 경로",
            value=kakao_default,
            key="kakao_base_path",
        )
        kakao_exps = list_experiments(Path(kakao_base))
        kakao_exp = st.selectbox("카카오 실험", kakao_exps, key="kakao_exp") if kakao_exps else ""
        kakao_exp_base = resolve_coord_base(Path(kakao_base), kakao_exp, "kakao") if kakao_exp else Path()
        kakao_coords = list_coord_folders(kakao_exp_base) if kakao_exp else []
        kakao_coord = st.selectbox("카카오 좌표 폴더", kakao_coords, key="kakao_coord") if kakao_coords else ""
        kakao_routes_dir = resolve_routes_dir(
            kakao_exp_base,
            kakao_coord,
            route_type,
            "kakao",
        )

    with st.sidebar.expander("OSRM 소스", expanded=True):
        osrm_default = st.session_state.get("last_osrm_base", str(default_osrm_base))
        if sync_base:
            osrm_default = str(default_osrm_base)
        osrm_base = st.text_input(
            "OSRM 기본 경로",
            value=osrm_default,
            key="osrm_base_path",
        )
        osrm_exps = list_experiments(Path(osrm_base))
        default_osrm_exp = st.session_state.get("last_osrm_exp")
        if default_osrm_exp in osrm_exps:
            osrm_exp_index = osrm_exps.index(default_osrm_exp)
        else:
            osrm_exp_index = 0
        osrm_exp = st.selectbox(
            "OSRM 실험",
            osrm_exps,
            index=osrm_exp_index if osrm_exps else 0,
            key="osrm_exp",
        ) if osrm_exps else ""
        osrm_exp_base = resolve_coord_base(Path(osrm_base), osrm_exp, "osrm") if osrm_exp else Path()
        osrm_coords = list_coord_folders(osrm_exp_base) if osrm_exp else []
        default_osrm_coord = st.session_state.get("last_osrm_coord")
        if default_osrm_coord in osrm_coords:
            osrm_coord_index = osrm_coords.index(default_osrm_coord)
        else:
            osrm_coord_index = 0
        osrm_coord = st.selectbox(
            "OSRM 좌표 폴더",
            osrm_coords,
            index=osrm_coord_index if osrm_coords else 0,
            key="osrm_coord",
        ) if osrm_coords else ""
        osrm_routes_dir = resolve_routes_dir(
            osrm_exp_base,
            osrm_coord,
            route_type,
            "osrm",
        )

    kakao_routes = load_routes(kakao_routes_dir).get("kakao", {})
    osrm_routes = load_routes(osrm_routes_dir).get("osrm", {})

    if not kakao_routes and not osrm_routes:
        st.warning("경로 JSON 파일을 찾지 못했습니다. routes 폴더를 확인하세요.")
        return

    kakao_indices = sorted(idx for idx in kakao_routes.keys() if idx is not None)
    osrm_indices = sorted(idx for idx in osrm_routes.keys() if idx is not None)

    kakao_hospital_map = {}
    osrm_hospital_map = {}
    if route_type == "hos2site":
        if kakao_coord:
            kakao_hospital_map = load_hospital_type_map(kakao_exp_base, kakao_coord)
        if osrm_coord:
            osrm_hospital_map = load_hospital_type_map(osrm_exp_base, osrm_coord)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("카카오")
        if not kakao_indices:
            st.warning("카카오 경로가 없습니다.")
        else:
            kakao_idx = st.selectbox(
                "카카오 경로 인덱스",
                kakao_indices,
                format_func=lambda idx: format_route_label(kakao_routes, idx),
                key="kakao_route_index",
            )
            kakao_record = kakao_routes.get(kakao_idx)
            if not kakao_record:
                st.warning("해당 인덱스의 카카오 경로가 없습니다.")
            else:
                meta = kakao_record.get("meta", {})
                st.write(
                    f"거리: {meta.get('distance_km', 'n/a')} km | 시간: {meta.get('duration_min', 'n/a')} 분"
                )
                m, err = make_kakao_map(kakao_record, hospital_type_map=kakao_hospital_map)
                if err:
                    st.error(err)
                else:
                    st_folium(m, use_container_width=True, height=500)
                st.caption("교통 색상: 초록=원활, 노랑=서행, 빨강=혼잡, 회색=정보 없음.")

    with col_right:
        st.subheader("OSRM")
        if not osrm_indices:
            st.warning("OSRM 경로가 없습니다.")
        else:
            osrm_idx = st.selectbox(
                "OSRM 경로 인덱스",
                osrm_indices,
                format_func=lambda idx: format_route_label(osrm_routes, idx),
                key="osrm_route_index",
            )
            osrm_record = osrm_routes.get(osrm_idx)
            if not osrm_record:
                st.warning("해당 인덱스의 OSRM 경로가 없습니다.")
            else:
                meta = osrm_record.get("meta", {})
                st.write(
                    f"거리: {meta.get('distance_km', 'n/a')} km | 시간: {meta.get('duration_min', 'n/a')} 분"
                )
                m, err = make_osrm_map(osrm_record, hospital_type_map=osrm_hospital_map)
                if err:
                    st.error(err)
                else:
                    st_folium(m, use_container_width=True, height=500)
                st.caption("교통 정보 미반영 상태의 고정 데이터입니다.")


def main():
    st.set_page_config(page_title="경로 비교", layout="wide")
    if "page_choice" not in st.session_state:
        st.session_state.page_choice = "생성"
    if "next_page" in st.session_state:
        st.session_state.page_choice = st.session_state.pop("next_page")

    page = st.sidebar.radio("페이지", ["생성", "시각화"], key="page_choice")
    if page == "생성":
        render_generate()
    else:
        render_visualize()


if __name__ == "__main__":
    main()
