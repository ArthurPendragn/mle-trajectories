// The theme choice lives in localStorage and on <html data-theme>, which
// globals.css reads; with none stored the page follows the system. THEME_SCRIPT
// runs inline in <head> and applies a stored choice before first paint, so a
// pinned theme never flashes. A plain module, not the "use client" toggle: a
// server component importing a value from a client module gets a reference,
// not the string.
export const THEME_KEY = "theme";

export const THEME_SCRIPT = `try{var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}catch(e){}`;
