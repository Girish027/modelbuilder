buildSuccess = true
Master_Build_ID="1.0.0"
grpId = "com.tfs.learning.systems"
artId = "modelbuilder-worker"
packageType = "zip"
artifactName = "${artId}-${Master_Build_ID}.${packageType}"
mavenVersion="apache-maven-3.3.3"
nodejsVersion="node-v4.4.6-linux-x64"
grailsVersion="grails-2.5.0"
gradleVersion="gradle-2.3"
pythonVersion="python3.6"
pythonPath="/var/tellme/jenkins/python3.6/bin"
python="${pythonPath}/${pythonVersion}"

isSanity = false
node("jenkins-slave03.pool.sv2.247-inc.net") {
	//********************************************************************#############**************************************************
	env.JAVA_HOME = "${env.jdk7_home}"
	sh "${env.JAVA_HOME}/bin/java -version"
	echo "Current branch <${env.BRANCH_NAME}>"
	def workspace = env.WORKSPACE

	stage('Clean') {
		cleanWs()
	}
	stage('Preparation') { 
		executeCheckout()
	}
	
	if(env.CHANGE_ID) {
		stage('commit') {
			echo "pull request detected"
			//Run the maven build
			// def result = true;
			// buildSuccess = executeBuild()
			// echo "buildSuccess = ${buildSuccess}"
			// validateBuild(buildSuccess)
			currentBuild.result = 'SUCCESS'
		}
		
	}
	
	if(!env.CHANGE_ID) {
		stage('sanity') {
			echo "push detected"
			def result = true;
			buildSuccess = executeBuildsanity()
			echo "buildSuccess = ${buildSuccess}"
			validateBuild(buildSuccess)
		}		
	}
}

def boolean executeBuild() {
	def result = true
	def branchName = env.BRANCH_NAME
	echo "branch = ${branchName}"
	Master_Build_ID = Master_Build_ID + "-" + branchName + "-" + env.BUILD_ID
	currentBuild.displayName = Master_Build_ID;

	try {
		sh ''' 	export PATH=${jdk8_home}/bin:$PATH
						mavenVersion='''+mavenVersion+'''
						nodejsVersion='''+nodejsVersion+'''
						grailsVersion='''+grailsVersion+'''
						gradleVersion='''+gradleVersion+'''
                        python='''+python+'''
						pythonPath='''+pythonPath+'''
						
						export PATH=$PATH:/opt/${mavenVersion}/bin
						export PATH=/opt/${nodejsVersion}/bin:$PATH
						export PATH=/var/tellme/jenkins/tools/sbt/bin:$PATH
						export PATH=/opt/${grailsVersion}/bin:$PATH
						export PATH=/opt/${gradleVersion}/bin:$PATH
						BRANCH='''+branchName+'''

						REPO_URL=${NEXUS_REPO_URL_SANITY}
						REPO_ID=${NEXUS_REPO_ID_SANITY}
						GRP_ID='''+grpId+'''
						ART_ID='''+artId+'''
						PACKAGE_TYPE='''+packageType+'''
						ARTIFACT_NAME='''+artifactName+'''
						Master_Build_ID='''+Master_Build_ID+'''


				WS_LOCATION=$(pwd)
export http_proxy="http://proxy-grp1.lb-priv.sv2.247-inc.net:3128"
export https_proxy="http://proxy-grp1.lb-priv.sv2.247-inc.net:3128"
${python} -m venv /var/tellme/modelbuilder_worker/orionEnv
source /var/tellme/modelbuilder_worker/orionEnv/bin/activate
pip install --no-cache-dir -r requirement.txt
mv /var/tellme/modelbuilder_worker/orionEnv .
rm -rf modelbuilder-worker.zip
zip -r modelbuilder-worker.zip *
/opt/${mavenVersion}/bin/mvn -B deploy:deploy-file -Durl=$REPO_URL -DrepositoryId=$REPO_ID -DgroupId=$GRP_ID -DartifactId=$ART_ID -Dversion=1.0.33 -Dpackaging=$PACKAGE_TYPE -Dfile=${WORKSPACE}/modelbuilder-worker.zip -DgeneratePom=true -e
/opt/${mavenVersion}/bin/mvn -B deploy:deploy-file -Durl=$REPO_URL -DrepositoryId=$REPO_ID -DgroupId=$GRP_ID -DartifactId=$ART_ID -Dversion=promoted -Dpackaging=$PACKAGE_TYPE -Dfile=${WORKSPACE}/modelbuilder-worker.zip -DgeneratePom=true -e
'''
		echo "Build Success...."
		result = true
	} catch(Exception ex) {
			echo "Build Failed...."
			echo "ex.toString() - ${ex.toString()}"
			echo "ex.getMessage() - ${ex.getMessage()}"
			echo "ex.getStackTrace() - ${ex.getStackTrace()}"
			result = false
	} 
		
	
	echo "result - ${result}"
	result
}
def executeBuildsanity() {
	def result = true
	def branchName = env.BRANCH_NAME
	echo "branch = ${branchName}"
	Master_Build_ID = Master_Build_ID + "-" + branchName + "-" + env.BUILD_ID
	currentBuild.displayName = Master_Build_ID;
	try {
		// slackSend channel: '#ml-workbench-dev', message: "Started ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Open>)"
		sh ''' 	export JAVA_HOME=${jdk8_home}
						export PATH=${jdk8_home}/bin:$PATH
						mavenVersion='''+mavenVersion+'''
						nodejsVersion='''+nodejsVersion+'''
						grailsVersion='''+grailsVersion+'''
						gradleVersion='''+gradleVersion+'''
                        python='''+python+'''
						pythonPath='''+pythonPath+'''
						
						export PATH=$PATH:/opt/${mavenVersion}/bin
						export PATH=/opt/${nodejsVersion}/bin:$PATH
						export PATH=/var/tellme/jenkins/tools/sbt/bin:$PATH
						export PATH=/opt/${grailsVersion}/bin:$PATH
						export PATH=/opt/${gradleVersion}/bin:$PATH
						BRANCH='''+branchName+'''

						REPO_URL=${NEXUS_REPO_URL_SANITY}
						REPO_ID=${NEXUS_REPO_ID_SANITY}
						GRP_ID='''+grpId+'''
						ART_ID='''+artId+'''
						PACKAGE_TYPE='''+packageType+'''
						ARTIFACT_NAME='''+artifactName+'''
						Master_Build_ID='''+Master_Build_ID+'''

						WS_LOCATION=$(pwd)

						
export http_proxy="http://proxy-grp1.lb-priv.sv2.247-inc.net:3128"
export https_proxy="http://proxy-grp1.lb-priv.sv2.247-inc.net:3128"
${python} -m venv /var/tellme/modelbuilder_worker/orionEnv
source /var/tellme/modelbuilder_worker/orionEnv/bin/activate
pip install --no-cache-dir -r requirement.txt
mv /var/tellme/modelbuilder_worker/orionEnv .
rm -rf modelbuilder-worker.zip
zip -r modelbuilder-worker.zip *
/opt/${mavenVersion}/bin/mvn -B deploy:deploy-file -Durl=$REPO_URL -DrepositoryId=$REPO_ID -DgroupId=$GRP_ID -DartifactId=$ART_ID -Dversion=1.0.33 -Dpackaging=$PACKAGE_TYPE -Dfile=${WORKSPACE}/modelbuilder-worker.zip -DgeneratePom=true -e
/opt/${mavenVersion}/bin/mvn -B deploy:deploy-file -Durl=$REPO_URL -DrepositoryId=$REPO_ID -DgroupId=$GRP_ID -DartifactId=$ART_ID -Dversion=promoted -Dpackaging=$PACKAGE_TYPE -Dfile=${WORKSPACE}/modelbuilder-worker.zip -DgeneratePom=true -e					
'''
		echo "Build Success...."
// slackSend channel: '#ml-workbench-dev', message: "Finished ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Close>)"
		result = true
	} catch(Exception ex) {
		 echo "Build Failed...."
		 echo "ex.toString() - ${ex.toString()}"
		 echo "ex.getMessage() - ${ex.getMessage()}"
		 echo "ex.getStackTrace() - ${ex.getStackTrace()}"
		 // slackSend channel: '#ml-workbench-dev', message: "Failed ${env.JOB_NAME} ${env.BUILD_NUMBER} (<${env.BUILD_URL}|Close>)"
		 result = false
	} 
	result 
}
def executeCheckout() {
  //Get code from GitHub repository
  checkout scm
}

def validateBuild(def buildStatus) {
	if (buildStatus) {
		currentBuild.result = 'SUCCESS'
	} else {
		currentBuild.result = 'FAILURE'
		error "build failed!"
	}
}

