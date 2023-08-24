#version 450
#extension GL_KHR_vulkan_glsl:enable

//specify the precision of floating point
precision highp float;

layout(location = 0) in vec2 texCoords;
layout(location = 1) in vec3 worldPos;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 tangents;
layout(location = 4) in vec4 lightSpacePos;

layout(push_constant) uniform PushConstantData {
    vec4 cameraPos;
} pushConstant;

layout(set = 1,binding = 0) uniform sampler2D albedoMap;
layout(set = 1,binding = 1) uniform sampler2D metallicMap;
layout(set = 1,binding = 2) uniform sampler2D roughnessMap;   
layout(set = 1,binding = 3) uniform sampler2D normalMap;
layout(set = 1,binding = 4) uniform sampler2D aoMap;  

layout(set = 2,binding = 0) uniform ULight{
	vec4 position;
    vec4 direction;
	vec4 color;
    float cutoff;
}uLight;

layout(set = 3,binding = 0) uniform sampler2D shadowMap;

const float PI = 3.14159265359;

layout(location = 0) out vec4 oColor;

vec3 getNormalFromMap()
{
    vec3 tangentNormal = (texture(normalMap, texCoords).rgb) * 2.0 - 1.0;

	vec3 N = normalize(normal);    
    vec3 T = normalize(tangents);   
    vec3 B = cross(N, T);                 
    mat3 TBN = mat3(T, B, N);                 

    return normalize( tangentNormal);
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
	float a = roughness*roughness;
	float a2 = a*a;
	float NdotH = max(dot(N, H), 0.0);
	float NdotH2 = NdotH*NdotH;
	float denom = (NdotH2 * (a2 - 1.0) + 1.0);
	denom = PI * denom * denom;
	return a2 / denom;
}

vec3 fresnelSchlick(float cosTheta, vec3 F0)
{
	return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

#define ambient 0.1
float getShadow(vec4 shadowCoord,vec2 off)
{
	float shadow = 1.0;
    float bias = 0.01;
	if ( shadowCoord.z > -1.0 && shadowCoord.z < 1.0 ) 
	{
        vec3 projCoords = shadowCoord.xyz / shadowCoord.w;
        projCoords.y += off.y;
        projCoords.x += off.x;

		float dist = texture(shadowMap, vec2(shadowCoord.st + off)).r;
		if ( shadowCoord.w > 0.0 && dist < shadowCoord.z - bias ) 
		{
			shadow = ambient;
		}
	}
	return shadow;
}

float PCF(vec4 lightSpacePos)
{
	ivec2 texDim = textureSize(shadowMap, 0).xy;
	float scale = 0.75;
	float dx = scale * 1.0 / float(texDim.x);
	float dy = scale * 1.0 / float(texDim.y);

	float shadowFactor = 0.0;
	int count = 0;
	int range = 1;
	
	for (int x = -range; x <= range; x++) {
		for (int y = -range; y <= range; y++) {
			shadowFactor += getShadow(lightSpacePos, vec2(dx*x, dy*y));
			count++;
		}
	}
	return shadowFactor / count;
}

void main()
{
    vec3  albedo    =  texture(albedoMap, texCoords).rgb;
    float metallic  = texture(metallicMap, texCoords).r;

    float roughness = texture(roughnessMap, texCoords).r;

    float depth = texture(shadowMap, texCoords).r;

    vec4 shadowCoord = lightSpacePos / lightSpacePos.w;
    shadowCoord.y = 1.0 - shadowCoord.y;

    if(texture(albedoMap, texCoords).a < 0.1)
        discard;
    else{
    float shadow = PCF(lightSpacePos / lightSpacePos.w);
    //float shadow = getShadow(lightSpacePos / lightSpacePos.w,vec2(0.f,0.f));


    vec3 n = getNormalFromMap();

    // view direction, point to the camera
    vec3 v = normalize(vec3(pushConstant.cameraPos) - worldPos);
    vec3 l = normalize(vec3(uLight.position) - worldPos);
    vec3 h = normalize(v + l);  

    float ndoth = clamp(dot(n, h), 0.0 ,1.0);
    float ndotv = clamp(dot(n, v), 0.0 ,1.0);
    float ndotl = clamp(dot(n, l), 0.0 ,1.0);
    float ldoth = clamp(dot(l, h), 0.0, 1.0);
    float vdoth = dot(v, h);
    
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    vec3 LAmbient = albedo ;

    vec3 color = vec3(0.0);

    //float theta = dot(l, normalize(vec3(-uLight.direction)));
    
    vec3 F = fresnelSchlick(vdoth,F0);
    //Fresnel factor

    vec3 Ldiffuse = ( albedo / PI ) * ( vec3(1) - F ) * ( 1 - metallic );

	float Dh =  DistributionGGX(n,h,roughness);
    //distribution
   
    float Glv = min( 1 , min( 2 * ( ndoth * ndotv ) / vdoth , 2 * ( ndoth * ndotl)  / vdoth ) );
   
    vec3 specular = F * Dh * Glv / max(0.000001, 4.0 * ndotl * ndotv);
   
    color += (specular + Ldiffuse) * ndotl * shadow + LAmbient * 0.001 ;

    float luminance = dot(color, vec3(0.2126, 0.7152, 0.0722));
	vec3 scaledColor = color / (luminance + 1.0);
    vec3 mappedColor = scaledColor / (scaledColor + 1.0);
    oColor = vec4(color * shadow, 1.0);
    }
}