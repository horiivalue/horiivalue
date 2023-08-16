<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>基本</title>
</head>
<body>
世界に１人 かけがえのないあなた<br>
<p>何かしゃべって!</p>
<form>
<input type="text" name="a">
<input type="submit" value="送信するよ！">
</form>
<?php
print isset($_GET["a"])?$_GET["a"]."でござる":"何かしゃべって！";
?>
</body>
</html>