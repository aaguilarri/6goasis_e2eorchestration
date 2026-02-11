/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// CarlaAppPlacementSpec defines the desired state of CarlaAppPlacement.
type CarlaAppPlacementSpec struct {
	// INSERT ADDITIONAL SPEC FIELDS - desired state of cluster
	// Important: Run "make" to regenerate code after modifying this file

	// Foo is an example field of CarlaAppPlacement. Edit carlaappplacement_types.go to remove/update
	Foo        string    `json:"foo,omitempty"`
	AppID      string    `json:"appId"`
	EdgeNodeID string    `json:"edgeNodeId"`
	Kafka      KafkaSpec `json:"kafka"`
}

type KafkaSpec struct {
	Broker  string       `json:"broker"`
	Topic   string       `json:"topic"`
	Message KafkaMessage `json:"message"`
}

type KafkaMessage struct {
	EventType string `json:"eventType"`
}

// CarlaAppPlacementStatus defines the observed state of CarlaAppPlacement.
type CarlaAppPlacementStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
	LastNotifiedEdgeNodeID string `json:"lastNotifiedEdgeNodeId,omitempty"`
	LastNotificationTime   string `json:"lastNotificationTime,omitempty"`
	KafkaNotificationSent  bool   `json:"kafkaNotificationSent,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// CarlaAppPlacement is the Schema for the carlaappplacements API.
type CarlaAppPlacement struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   CarlaAppPlacementSpec   `json:"spec,omitempty"`
	Status CarlaAppPlacementStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// CarlaAppPlacementList contains a list of CarlaAppPlacement.
type CarlaAppPlacementList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []CarlaAppPlacement `json:"items"`
}

func init() {
	SchemeBuilder.Register(&CarlaAppPlacement{}, &CarlaAppPlacementList{})
}
