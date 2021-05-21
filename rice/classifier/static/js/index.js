var object = {
    nitrogen: document.getElementById("Nitrogen"),
    phosporus: document.getElementById("Phosphorus"),
    pottasium: document.getElementById('Pottasium'),
    temperature: document.getElementById('Temperature'),
    ph: document.getElementById('Ph'),
    humidity: document.getElementById('Humidity'),
    rainfall: document.getElementById('Rainfall'),
    myOffcanvas: document.getElementById('offcanvasWithBothOptions'),
    pred: document.getElementById('pred'),
    Predict: $("#Predict"),
    toast: document.getElementById("toast"),
}

checkString = (str) => {
    return (str === "" || str === " ");
}



object.Predict.submit((event) => {
    event.preventDefault();
    if (checkString(object.nitrogen.value) || checkString(object.phosporus.value) || checkString(object.pottasium.value) || checkString(object.temperature.value) || checkString(object.humidity.value) || checkString(object.ph.value) || checkString(object.rainfall.value)) {
        var toast = new bootstrap.Toast(object.toast);
        toast.show();
        return;
    }
    var bsOffcanvas = new bootstrap.Offcanvas(object.myOffcanvas);
    bsOffcanvas.show();
    predict();
})


console.log(object.nitrogen.value);