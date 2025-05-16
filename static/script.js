function takepic() {
	fetch("v1/savepic")
		.then((r) => r.json())
		.then(t => {
			e = document.getElementById("savepic-status")
			console.log(t)
			e.innerText = t.answer
		})
}
function calibrate() {
	fetch("v1/calibrate")
}
function zoomout() {
	fetch("v1/zoomout")
}

function commit() {
	1+1
}

/*inputs = document.getElementsByTagName("input")
for (let i = 0; i < inputs.length; i++)
	c = inputs[i];
	c.autocomplete = "off";
	c.spellcheck = "false";
	c.autocomplete = "off";
*/
function updatePodium(prefix, res) {
	json = JSON.parse(res[prefix + "-podium"]);
	for (let i = 1; i < 4; i++) {
		el = document.getElementById(prefix + "-" + i)
		el.getElementsByTagName("p")[0].innerText = json["top"][i-1];
		el.getElementsByClassName("podium-rank")[0].innerHTML = json["mid"][i-1] + "<br/>(" + json["bot"][i-1] + ")";
	}
}

/*document.getElementById("roundForm").addEventListener("submit", function (e) {
	e.preventDefault();
	
	form = document.getElementById("roundForm")
	var formData = new FormData(form);
	// output as an object
	console.log(Object.fromEntries(formData));
	
	fetch("/enter_game", {
		method: "POST",
		/*headers: {
			"Accept": "application/json",
			"Content-Type": "application/json"
		},
		body: formData
	})
		.then((res) => res.json())
		.then((e) => {
			console.log(e)
			document.getElementById("table-single").innerHTML = e["single-board"]
			document.getElementById("table-team").innerHTML = e["team-board"]

			updatePodium("single", e);
			updatePodium("team", e);

		})
});*/

// press enter to take picture
document.onkeydown = function(e){
    //e = e || window.event;
    var key = e.keyCode;
	console.log(e)
    if(key===13){
        takepic();
    }
}

// every 55 seconds ping the server to say "hey there, I am still connected, please keep rendering frame :)"
// if there where no pings in 60s, frame generation stops.
setInterval(() => {fetch("website/liveline");}, 55000) 

function testAPI() {
	fetch("/api-doc")
		.then((r) => r.json())
		.then((a) => a.api)
		.then((api) => {
			console.log(api)
			di = document.getElementById("api-doc")
			for (let i=0; i < api.length; i++) {
				//console.log(api[i])
				var button = document.createElement("button");
				button.innerText = api[i];
				button.addEventListener("click", () => {
					fetch(api[i].substring(1))
						.then((r) => {
							try{
								return r.json()
							} catch (error) {
								console.log(error, r)
							}})
						.then((r) => {
							document.getElementById("api-out").innerText = api[i] + ": " + JSON.stringify(r,null, 2);
							console.log(r, api[i])
						})
				});
				di.appendChild(button)
			}
		})
}

testAPI()