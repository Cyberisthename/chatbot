var styles = document.getElementsByTagName('style');
var headElements = document.getElementsByTagName('head');
if (headElements && headElements.length > 0) {
	var head = headElements[0];
	var StylesContent = [];
	for (var i = 0; styles.length; i++) {
		StylesContent.push(styles[0].textContent);
		styles[0].parentElement.removeChild(styles[0]);
	}

	for (var i = 0; i < StylesContent.length; i++) {
		var style = document.createElement('style');
		style.type = 'text/css';
		if (style.styleSheet) {
			style.styleSheet.css = StylesContent[i];
		}
		else {
			style.appendChild(document.createTextNode(StylesContent[i]));
		}
		head.appendChild(style);
	}
}