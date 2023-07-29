<?php
/**
 * @nick 慕名
 * @author 慕名API
 * @link http://xiaoapi.cn/
 * @date 2022年10月17日13:25:19
 * @msg 小爱ai聊天，返回txt和tts语音链接可返回文本和json。
 * @n 4
 * @Version 1.0
**/
require "./function.php"; // 引入函数文件
addApiAccess(54); // 调用统计函数
Header('Content-Type:application/json');

$txt=$_REQUEST["msg"];

if(!$txt){
    $json['code']=201;
    $json['data']['msg']="抱歉，输入不能为空。";
    $json['data']['tips']="慕名API：http://xiaoapi.cn";
    echo ret_json($json);
    die;

}
$ua="Mozilla/5.0 (Linux; Android 4.4.4; vivo Y13L Build/KTU84P) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/33.0.0.0 Mobile Safari/537.36";
$json1=json_decode(get_curl("https://ai-voice.api.xiaomi.net/aivs/v2.2/text?erequestId=&token=&userId=de5eb5e5-11ba-4a3a-8b34-4e5f4b956a10&latitude=&longitude=","{requestText:".$txt."}",0,0,0,$ua), true);
$txt=$json1["responseText"];
if(!$txt){
    $txt=$json1["directive"]["displayText"];
}
if($_REQUEST["type"]=="txt"){

    if(!$txt){
        echo "抱歉，获取不到数据。";
    }else{
        echo $txt;
    }
    die;
}
if(!$txt){
    $json['code']=201;
    $json['data']['msg']="抱歉，获取不到数据。";
    $json['data']['tips']="慕名API：http://xiaoapi.cn";
    echo ret_json($json);

}else{
    $json['code']=200;
    $json['data']['txt']=$txt;
    $json['data']['tts']=$json1["directive"]["url"];
    $json['data']['tips']="慕名API：http://xiaoapi.cn";
    echo ret_json($json);

}




function ret_json($json){
    return stripslashes(json_encode($json,JSON_UNESCAPED_UNICODE | JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES));
}


function get_curl($url,$post=0,$referer=1,$cookie=0,$header=0,$ua=0,$nobaody=0)
{
$ch = curl_init();
curl_setopt($ch, CURLOPT_URL,$url);
curl_setopt($ch, CURLOPT_SSL_VERIFYPEER, false);
curl_setopt($ch, CURLOPT_SSL_VERIFYHOST, false);
$httpheader[] = "Accept:application/json";
$httpheader[] = "Accept-Encoding:gzip,deflate,sdch";
$httpheader[] = "Accept-Language:zh-CN,zh;q=0.8";
$httpheader[] = "Connection:close";
curl_setopt($ch, CURLOPT_HTTPHEADER, $httpheader);
if($post){
curl_setopt($ch, CURLOPT_POST, 1);
curl_setopt($ch, CURLOPT_POSTFIELDS, $post);
}
if($header){
curl_setopt($ch, CURLOPT_HEADER, TRUE);
}
if($cookie){
curl_setopt($ch, CURLOPT_COOKIE, $cookie);
}
if($referer){
if($referer==1){
curl_setopt($ch, CURLOPT_REFERER, 'http://m.qzone.com/infocenter?g_f=');
}else{
curl_setopt($ch, CURLOPT_REFERER, $referer);
}
}
if($ua){
curl_setopt($ch, CURLOPT_USERAGENT,$ua);
}else{
curl_setopt($ch, CURLOPT_USERAGENT,'Mozilla/5.0 (Linux; U; Android 4.4.1; zh-cn) AppleWebKit/533.1 (KHTML, like Gecko)Version/4.0 MQQBrowser/5.5 Mobile Safari/533.1');
}
if($nobaody){
curl_setopt($ch, CURLOPT_NOBODY,1);
}
curl_setopt($ch, CURLOPT_ENCODING, "gzip");
curl_setopt($ch, CURLOPT_RETURNTRANSFER,1);
$ret = curl_exec($ch);
curl_close($ch);
return $ret;
}

?>