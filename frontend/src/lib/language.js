const ARABIC_PATTERN = /[\u0600-\u06ff]/;

export function messageLanguage(text = "") {
  return ARABIC_PATTERN.test(text) ? "ar" : "en";
}

export function messageDirection(text = "") {
  return messageLanguage(text) === "ar" ? "rtl" : "ltr";
}
