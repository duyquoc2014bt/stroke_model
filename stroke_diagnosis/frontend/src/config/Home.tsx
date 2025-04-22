import { DefaultOptionType } from "antd/es/select";

export const selectKeys = [
  "family_factor",
  "diabetes_mellitus",
  "hyperlipidemia",
  "smoking_status",
  "pre_stroke",
  "face_weakness",
  "arm_weakness",
  "speech_problem",
  "headache",
  "dizziness_nausea",
];

export const selectionFields = selectKeys.map((m) => {
  const handleString = m.replace(/_/g, " ");
  return {
    label: handleString.charAt(0).toUpperCase() + handleString.slice(1),
    name: m,
  };
});

export const commonOptions: DefaultOptionType[] = [
  {
    label: "Yes",
    value: 1,
  },
  {
    label: "No",
    value: 0,
  },
];
