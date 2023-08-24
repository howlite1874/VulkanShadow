#version 450
#extension GL_EXT_debug_printf : enable
layout(location = 0) in vec3 iPosition;
layout(location = 1) in vec2 iTexcoord;
layout(location = 2) in vec3 iNormals;
layout(location = 3) in vec3 iTangents;

//std140
layout(set = 0,binding = 0) uniform UScene{
	mat4 camera;
	mat4 projection;
	mat4 projCamera;
	mat4 lightSpaceMatrix;
}uScene;

const mat4 biasMat = mat4( 
	0.5, 0.0, 0.0, 0.0,
	0.0, 0.5, 0.0, 0.0,
	0.0, 0.0, 1.0, 0.0,
	0.5, 0.5, 0.0, 1.0 );

layout(location = 0) out vec2 texCoords;
layout(location = 1) out vec3 worldPos;
layout(location = 2) out vec3 normal;
layout(location = 3) out vec3 tangent;
layout(location = 4) out vec4 lightSpacePos;

void main()
{
	texCoords = iTexcoord;
	worldPos = iPosition;
	normal = iNormals;
	tangent = iTangents;
	lightSpacePos =biasMat * uScene.lightSpaceMatrix * vec4(iPosition,1.f);
	gl_Position = uScene.projCamera * vec4(iPosition,1.f);

}

