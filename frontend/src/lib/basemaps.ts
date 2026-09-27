/* Keyless Esri basemaps (Gray Canvas + World Imagery), with reference labels. */
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services'
export const ESRI_ATTRIBUTION =
  'Tiles &copy; <a href="https://www.esri.com">Esri</a>, HERE, Garmin, Maxar, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

export type BasemapKind = 'map' | 'satellite'

export function basemap(kind: BasemapKind, theme: 'light' | 'dark') {
  if (kind === 'satellite') {
    return {
      base: `${ESRI}/World_Imagery/MapServer/tile/{z}/{y}/{x}`,
      labels: `${ESRI}/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}`,
      maxNative: 18,
    }
  }
  const v = theme === 'dark' ? 'Dark' : 'Light'
  return {
    base: `${ESRI}/Canvas/World_${v}_Gray_Base/MapServer/tile/{z}/{y}/{x}`,
    labels: `${ESRI}/Canvas/World_${v}_Gray_Reference/MapServer/tile/{z}/{y}/{x}`,
    maxNative: 16,   // Gray Canvas tiles stop at z16; Leaflet upsamples beyond
  }
}
