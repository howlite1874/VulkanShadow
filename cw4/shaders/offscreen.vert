#version 450 core

layout (location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform uMatrix
{
	mat4 lightSpaceMatrix;
}UMatrix;

void main()
{
	vec4 lightSpacePos = UMatrix.lightSpaceMatrix * vec4(inPosition, 1.0);
	gl_Position = lightSpacePos;
}

