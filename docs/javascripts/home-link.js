// Make the "SO3LR" site name in the header a link to the home page, like the
// logo next to it. The root URL is resolved once, from the logo's link on the
// first page loaded, because instant navigation keeps the header in place.
(function () {
  var home = null;

  function link() {
    var logo = document.querySelector(".md-header__button.md-logo");
    var name = document.querySelector(".md-header__title .md-header__topic:first-child .md-ellipsis");
    if (!logo || !name || name.tagName === "A") return;
    if (!home) home = new URL(logo.getAttribute("href"), location.href).href;
    var a = document.createElement("a");
    a.className = name.className + " so3lr-home-link";
    a.href = home;
    a.textContent = name.textContent.trim();
    name.replaceWith(a);
  }

  if (window.document$) document$.subscribe(link);
  else document.addEventListener("DOMContentLoaded", link);
})();
