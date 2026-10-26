"""지도 안에서만 동작하는 타일 선택·위성 전환 컨트롤."""
import folium
from branca.element import MacroElement, Template


class MapControls(MacroElement):
    _template = Template("""
    {% macro script(this, kwargs) %}
    (function() {
      const map = {{ this._parent.get_name() }};
      const layers = [{{ this.street }}, {{ this.gray }}];
      const satellite = {{ this.satellite }};
      let selected = 0;
      let imagery = false;
      try {
        selected = sessionStorage.getItem('mci-map-style') === '1' ? 1 : 0;
        imagery = sessionStorage.getItem('mci-map-imagery') === 'true';
      } catch (error) { /* 저장소를 차단한 브라우저에서는 현재 지도 안에서만 유지한다. */ }
      const control = L.control({position: 'topright'});
      control.onAdd = function() {
        const group = L.DomUtil.create('div', 'mci-map-styles leaflet-bar');
        group.setAttribute('role', 'group');
        group.setAttribute('aria-label', '지도 테마');
        ['도로지도', '회색지도'].forEach(function(label, index) {
          const button = L.DomUtil.create('button', 'mci-map-style', group);
          button.type = 'button';
          button.textContent = label;
          button.setAttribute('aria-pressed', String(index === selected));
          L.DomEvent.on(button, 'click', function() {
            selected = index;
            imagery = false;
            update();
          });
        });
        L.DomEvent.disableClickPropagation(group);
        L.DomEvent.disableScrollPropagation(group);
        return group;
      };
      control.addTo(map);
      const zoom = map.zoomControl.getContainer();
      const toggle = L.DomUtil.create('button', 'mci-satellite', zoom);
      toggle.type = 'button';
      toggle.textContent = '위성';
      toggle.title = '위성지도 전환';
      toggle.setAttribute('aria-label', '위성지도 전환');
      toggle.setAttribute('aria-pressed', 'false');
      L.DomEvent.on(toggle, 'click', function(event) {
        L.DomEvent.stop(event);
        imagery = !imagery;
        update();
      });
      function update() {
        layers.concat([satellite]).forEach(function(layer) {
          if (map.hasLayer(layer)) map.removeLayer(layer);
        });
        (imagery ? satellite : layers[selected]).addTo(map);
        control.getContainer().querySelectorAll('button').forEach(function(button, index) {
          button.setAttribute('aria-pressed', String(!imagery && index === selected));
        });
        toggle.setAttribute('aria-pressed', String(imagery));
        toggle.textContent = imagery ? '지도' : '위성';
        toggle.title = imagery ? '일반지도로 돌아가기' : '위성지도 전환';
        try {
          sessionStorage.setItem('mci-map-style', String(selected));
          sessionStorage.setItem('mci-map-imagery', String(imagery));
        } catch (error) { /* 타일 전환은 저장소 없이도 동작한다. */ }
      }
      update();
    })();
    {% endmacro %}
    """)

    def __init__(self, street, gray, satellite):
        super().__init__()
        self._name = "MCIMapControls"
        self.street = street.get_name()
        self.gray = gray.get_name()
        self.satellite = satellite.get_name()


def create_map(center, zoom=12):
    m = folium.Map(location=center, zoom_start=zoom, control_scale=True,
                   tiles=None, prefer_canvas=True)
    street = folium.TileLayer("OpenStreetMap", name="도로지도").add_to(m)
    gray = folium.TileLayer("Esri.WorldGrayCanvas", name="회색지도", show=False).add_to(m)
    satellite = folium.TileLayer("Esri.WorldImagery", name="위성지도", show=False).add_to(m)
    MapControls(street, gray, satellite).add_to(m)
    m.get_root().header.add_child(folium.Element("""<style>
      .leaflet-container { font-family: 'IBM Plex Sans KR', 'Malgun Gothic', sans-serif; }
      .leaflet-bar { box-shadow: 0 2px 8px #12212e22; border: 1px solid #cbd5df; }
      .mci-map-styles { display: flex; padding: 4px; gap: 4px; background: white; }
      .mci-map-style { min-width: 74px; min-height: 38px; padding: 8px 12px;
        border: 1px solid transparent; border-radius: 4px; color: #334b5d;
        background: #f0f4f7; font: inherit; font-size: 13px; cursor: pointer; }
      .mci-map-style[aria-pressed=true] { color: #075991; background: #e5f1fa; border-color: #0b63ce; }
      .mci-satellite { display: block; width: 30px; min-height: 38px; padding: 0;
        border: 0; border-top: 1px solid #ccc; background: white; color: #203c50;
        font: inherit; font-size: 11px; cursor: pointer; }
      .mci-satellite[aria-pressed=true] { background: #e5f1fa; color: #075991; }
      .mci-map-style:hover, .mci-satellite:hover { background: #dceaf4; }
      .mci-map-style:focus-visible, .mci-satellite:focus-visible { outline: 3px solid #0b63ce; outline-offset: 2px; }
      .leaflet-control-attribution { font-size: 10px; max-width: min(75vw, 560px); }
      .mci-legend { position: fixed; bottom: 28px; left: 12px; z-index: 999;
        background: #fffffff2; border: 1px solid #d4dfe7; border-radius: 6px;
        padding: 8px 12px; color: #263f50; font-size: 11px; line-height: 1.65; max-width: 260px; }
      .mci-legend summary { font-size: 12px; font-weight: 600; cursor: pointer; }
      @media (max-width: 600px) { .mci-legend { max-width: 200px; } }
    </style>"""))
    return m
