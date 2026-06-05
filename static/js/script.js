// SMOOTH FADE-IN ANIMATION
document.addEventListener("DOMContentLoaded", () => {

    const elements = document.querySelectorAll(".bg-gray-800");

    elements.forEach((el, index) => {
        el.style.opacity = 0;
        el.style.transform = "translateY(10px)";

        setTimeout(() => {
            el.style.transition = "0.5s ease";
            el.style.opacity = 1;
            el.style.transform = "translateY(0)";
        }, index * 100);
    });

});